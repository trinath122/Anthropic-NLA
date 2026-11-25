from __future__ import annotations

import json
from pathlib import Path

import pytest

from nla_audit.llm_client import (
    AllProvidersFailed,
    LLMClient,
    PermanentError,
    PolicyError,
    RateLimited,
)

CONFIG = Path(__file__).parents[1] / "configs" / "llm.yaml"
ENV = {"GROQ_API_KEY": "x", "GEMINI_API_KEY": "x", "OPENROUTER_API_KEY": "x"}
MSGS = [{"role": "user", "content": "hello"}]


class FakeTransport:
    """Scripted outcomes per model name; records every call."""

    def __init__(self, script: dict[str, list] | None = None) -> None:
        self.script = script or {}
        self.calls: list[str] = []

    def complete(self, provider, api_key, model, messages, params):
        self.calls.append(model)
        outcomes = self.script.get(model)
        if outcomes:
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return out, {"total_tokens": 1}
        return f"answer from {model}", {"total_tokens": 1}


def make_client(tmp_path, transport, env=ENV) -> LLMClient:
    return LLMClient.from_config(
        CONFIG, root=tmp_path, transport=transport, env=env, sleep=lambda s: None
    )


def test_shipped_config_satisfies_policy(tmp_path):
    make_client(tmp_path, FakeTransport())  # raises PolicyError if the config is unsafe


def test_cache_prevents_second_call(tmp_path):
    t = FakeTransport()
    c = make_client(tmp_path, t)
    a = c.chat("gen", MSGS)
    b = c.chat("gen", MSGS)
    assert a.text == b.text
    assert len(t.calls) == 1
    assert not a.cached and b.cached
    # survives a restart (on-disk)
    c2 = make_client(tmp_path, t)
    assert c2.chat("gen", MSGS).cached
    assert len(t.calls) == 1


def test_sample_index_is_part_of_key(tmp_path):
    t = FakeTransport()
    c = make_client(tmp_path, t)
    c.chat("gen", MSGS, temperature=1.0, sample_index=0)
    c.chat("gen", MSGS, temperature=1.0, sample_index=1)
    assert len(t.calls) == 2


def test_rate_limit_then_fallback(tmp_path):
    first = "openai/gpt-oss-120b"  # first in gen chain
    t = FakeTransport({first: [RateLimited("429")] * 10})
    c = make_client(tmp_path, t)
    r = c.chat("gen", MSGS)
    assert r.provider == "groq" and r.model == "qwen/qwen3.8-27b"
    assert t.calls.count(first) == 5  # max_attempts_per_model


def test_rate_limit_recovers_on_same_model(tmp_path):
    t = FakeTransport({"openai/gpt-oss-120b": [RateLimited("429", retry_after=1.0), "ok"]})
    r = make_client(tmp_path, t).chat("gen", MSGS)
    assert r.text == "ok" and r.model == "openai/gpt-oss-120b"


def test_permanent_error_skips_model(tmp_path):
    t = FakeTransport({"openai/gpt-oss-120b": [PermanentError("404 model not found")]})
    c = make_client(tmp_path, t)
    assert c.chat("gen", MSGS).model == "qwen/qwen3.8-27b"
    c.chat("gen", [{"role": "user", "content": "another"}])
    assert t.calls.count("openai/gpt-oss-120b") == 1  # marked dead, never retried


def test_missing_keys_skip_providers(tmp_path):
    t = FakeTransport()
    c = make_client(tmp_path, t, env={"OPENROUTER_API_KEY": "x"})
    assert c.chat("judge", MSGS).provider == "openrouter"


def test_all_fail(tmp_path):
    with pytest.raises(AllProvidersFailed):
        make_client(tmp_path, FakeTransport(), env={}).chat("gen", MSGS)


def test_ledger_records_provider_model_role(tmp_path):
    c = make_client(tmp_path, FakeTransport())
    c.chat("judge", MSGS, tag="unit")
    c.chat("judge", MSGS, tag="unit")
    rows = [
        json.loads(x) for x in (tmp_path / "cache" / "llm_ledger.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {r["role"] for r in rows} == {"judge"}
    assert rows[0]["provider"] == "gemini" and rows[0]["family"] == "gemini"
    assert [r["cached"] for r in rows] == [False, True]


def _write_cfg(tmp_path, gen_chain, judge_chain):
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["roles"] = {"gen": gen_chain, "judge": judge_chain}
    p = tmp_path / "llm.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_policy_blocks_gemini_as_training_data(tmp_path):
    p = _write_cfg(
        tmp_path, ["gemini/gemini-3.5-flash"], ["openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"]
    )
    with pytest.raises(PolicyError, match="licences"):
        LLMClient.from_config(p, root=tmp_path, transport=FakeTransport(), env=ENV)


def test_policy_blocks_same_family_judge(tmp_path):
    p = _write_cfg(tmp_path, ["groq/openai/gpt-oss-120b"], ["groq/openai/gpt-oss-20b"])
    with pytest.raises(PolicyError, match="independence"):
        LLMClient.from_config(p, root=tmp_path, transport=FakeTransport(), env=ENV)
