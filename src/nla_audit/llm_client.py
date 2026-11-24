"""One OpenAI-compatible client for every free LLM provider.

Guarantees (see configs/llm.yaml):
  * provider fallback along an ordered chain per *role* ("gen" or "judge");
  * rate limiting per provider + exponential backoff on 429 / 5xx;
  * an on-disk SQLite cache so an identical request is never sent twice;
  * licence policy: role "gen" (training data) only uses permissively licensed
    open-weight models; role "judge" must use a model family disjoint from "gen";
  * a JSONL ledger recording provider + model + role for every call.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

Message = dict[str, str]


# ─── errors ─────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base class."""


class PolicyError(LLMError):
    """The configured chains violate the licence / judge-independence policy."""


class RateLimited(LLMError):
    def __init__(self, msg: str, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class TransientError(LLMError):
    """5xx / timeout / connection reset: retry with backoff."""


class PermanentError(LLMError):
    """4xx other than 429 (bad key, unknown model, ...): skip this model."""


class AllProvidersFailed(LLMError):
    pass


# ─── config objects ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str
    requests_per_minute: float


@dataclass(frozen=True)
class ModelRef:
    key: str  # "provider/model-name" exactly as in configs/llm.yaml
    provider: str
    model: str
    family: str
    licence: str


@dataclass(frozen=True)
class RetrySpec:
    max_attempts_per_model: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 120.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    role: str
    provider: str
    model: str
    family: str
    cached: bool
    cache_key: str
    usage: dict[str, Any] = field(default_factory=dict)


def _split_key(key: str) -> tuple[str, str]:
    provider, _, model = key.partition("/")
    if not model:
        raise ValueError(f"model key {key!r} must look like 'provider/model'")
    return provider, model


def validate_policy(roles: Mapping[str, list[ModelRef]], gen_allowed_licences: set[str]) -> None:
    """Raise PolicyError if the role chains break amendments 4 and 5."""
    gen = roles.get("gen", [])
    judge = roles.get("judge", [])
    bad = [m.key for m in gen if m.licence not in gen_allowed_licences]
    if bad:
        raise PolicyError(
            f"role 'gen' produces training data and may only use licences "
            f"{sorted(gen_allowed_licences)}; offending models: {bad}"
        )
    overlap = {m.family for m in gen} & {m.family for m in judge}
    if overlap:
        raise PolicyError(
            f"judge independence violated: families {sorted(overlap)} appear in both "
            f"'gen' and 'judge' chains"
        )


# ─── cache + ledger ─────────────────────────────────────────────────────────


class LLMCache:
    """SQLite key → JSON value. Safe to share across threads in one process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "created REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._db.commit()

    def close(self) -> None:
        self._db.close()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def cache_key(model_key: str, messages: list[Message], params: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"model": model_key, "messages": messages, "params": dict(params)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─── transport ──────────────────────────────────────────────────────────────


class Transport(Protocol):
    def complete(
        self,
        provider: ProviderSpec,
        api_key: str,
        model: str,
        messages: list[Message],
        params: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]: ...


class OpenAICompatTransport:
    """Real network transport using the `openai` SDK (its own retries disabled)."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s
        self._clients: dict[tuple[str, str], Any] = {}

    def _client(self, provider: ProviderSpec, api_key: str) -> Any:
        import openai

        k = (provider.base_url, api_key)
        if k not in self._clients:
            self._clients[k] = openai.OpenAI(
                base_url=provider.base_url,
                api_key=api_key,
                max_retries=0,
                timeout=self.timeout_s,
            )
        return self._clients[k]

    def complete(
        self,
        provider: ProviderSpec,
        api_key: str,
        model: str,
        messages: list[Message],
        params: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        import openai

        client = self._client(provider, api_key)
        api_params = {k: v for k, v in params.items() if k != "sample_index"}
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **api_params)
        except openai.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            retry_after = float(ra) if ra and ra.replace(".", "").isdigit() else None
            raise RateLimited(str(e), retry_after) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise TransientError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientError(str(e)) from e
            raise PermanentError(f"{e.status_code}: {e}") from e
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return text, usage


# ─── client ─────────────────────────────────────────────────────────────────


class _RateLimiter:
    def __init__(self, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self._clock = clock
        self._sleep = sleep
        self._next_ok: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, provider: ProviderSpec) -> None:
        interval = 60.0 / max(provider.requests_per_minute, 1e-9)
        with self._lock:
            now = self._clock()
            start = max(now, self._next_ok.get(provider.name, now))
            self._next_ok[provider.name] = start + interval
        if start > now:
            self._sleep(start - now)


class LLMClient:
    def __init__(
        self,
        providers: Mapping[str, ProviderSpec],
        roles: Mapping[str, list[ModelRef]],
        gen_allowed_licences: set[str],
        cache: LLMCache,
        ledger: Ledger,
        retry: RetrySpec = RetrySpec(),
        transport: Transport | None = None,
        env: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_policy(roles, gen_allowed_licences)
        self.providers = dict(providers)
        self.roles = {r: list(chain) for r, chain in roles.items()}
        self.cache = cache
        self.ledger = ledger
        self.retry = retry
        self.transport = transport or OpenAICompatTransport()
        self.env = env if env is not None else os.environ
        self._sleep = sleep
        self._limiter = _RateLimiter(clock, sleep)
        self._dead_models: set[str] = set()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        root: str | Path | None = None,
        transport: Transport | None = None,
        env: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> LLMClient:
        path = Path(path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = Path(root) if root is not None else Path.cwd()
        providers = {
            name: ProviderSpec(
                name=name,
                base_url=p["base_url"],
                api_key_env=p["api_key_env"],
                requests_per_minute=float(p.get("requests_per_minute", 10)),
            )
            for name, p in cfg["providers"].items()
        }
        models: dict[str, ModelRef] = {}
        for key, m in cfg["models"].items():
            prov, name = _split_key(key)
            if prov not in providers:
                raise ValueError(f"model {key!r} references unknown provider {prov!r}")
            models[key] = ModelRef(key, prov, name, m["family"], m["licence"])
        roles: dict[str, list[ModelRef]] = {}
        for role, chain in cfg["roles"].items():
            missing = [k for k in chain if k not in models]
            if missing:
                raise ValueError(f"role {role!r} lists unknown models {missing}")
            roles[role] = [models[k] for k in chain]
        retry = RetrySpec(**cfg.get("retry", {}))
        return cls(
            providers=providers,
            roles=roles,
            gen_allowed_licences=set(cfg["gen_allowed_licences"]),
            cache=LLMCache(root / cfg["cache_path"]),
            ledger=Ledger(root / cfg["ledger_path"]),
            retry=retry,
            transport=transport,
            env=env,
            sleep=sleep,
        )

    # -- public API -----------------------------------------------------------
    def chat(
        self,
        role: str,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        sample_index: int = 0,
        tag: str | None = None,
        **extra: Any,
    ) -> LLMResponse:
        """Send one chat request for `role`, trying the role's chain in order.

        `sample_index` distinguishes intentional repeated samples of the same
        prompt (it is part of the cache key, never sent to the API).
        """
        if role not in self.roles:
            raise ValueError(f"unknown role {role!r}; configured: {sorted(self.roles)}")
        chain = self.roles[role]
        params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "sample_index": sample_index,
            **extra,
        }

        # 1) cache: any model in the chain that already answered wins (chain order).
        for m in chain:
            key = cache_key(m.key, messages, params)
            hit = self.cache.get(key)
            if hit is not None:
                return self._record(role, m, hit["text"], hit.get("usage", {}), key, True, tag)

        # 2) network, with fallback.
        errors: list[str] = []
        for m in chain:
            if m.key in self._dead_models:
                continue
            provider = self.providers[m.provider]
            api_key = self.env.get(provider.api_key_env, "")
            if not api_key:
                errors.append(f"{m.key}: no {provider.api_key_env}")
                continue
            key = cache_key(m.key, messages, params)
            try:
                text, usage = self._call_with_retry(provider, api_key, m, messages, params)
            except PermanentError as e:
                self._dead_models.add(m.key)
                errors.append(f"{m.key}: {e}")
                continue
            except (RateLimited, TransientError) as e:
                errors.append(f"{m.key}: gave up after retries: {e}")
                continue
            self.cache.put(key, {"text": text, "usage": usage, "model": m.key})
            return self._record(role, m, text, usage, key, False, tag)
        raise AllProvidersFailed(f"role {role!r}: every model failed: {errors}")

    # -- internals -------------------------------------------------------------
    def _call_with_retry(
        self,
        provider: ProviderSpec,
        api_key: str,
        m: ModelRef,
        messages: list[Message],
        params: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        last: Exception | None = None
        for attempt in range(self.retry.max_attempts_per_model):
            self._limiter.wait(provider)
            try:
                return self.transport.complete(provider, api_key, m.model, messages, params)
            except RateLimited as e:
                last = e
                delay = e.retry_after if e.retry_after is not None else self._backoff(attempt)
            except TransientError as e:
                last = e
                delay = self._backoff(attempt)
            if attempt + 1 < self.retry.max_attempts_per_model:
                self._sleep(min(delay, self.retry.max_delay_s))
        assert last is not None
        raise last

    def _backoff(self, attempt: int) -> float:
        return min(self.retry.base_delay_s * (2**attempt), self.retry.max_delay_s)

    def _record(
        self,
        role: str,
        m: ModelRef,
        text: str,
        usage: dict[str, Any],
        key: str,
        cached: bool,
        tag: str | None,
    ) -> LLMResponse:
        self.ledger.append(
            {
                "ts": time.time(),
                "role": role,
                "provider": m.provider,
                "model": m.model,
                "family": m.family,
                "licence": m.licence,
                "cached": cached,
                "cache_key": key,
                "tag": tag,
                "usage": usage,
            }
        )
        return LLMResponse(text, role, m.provider, m.model, m.family, cached, key, usage)
