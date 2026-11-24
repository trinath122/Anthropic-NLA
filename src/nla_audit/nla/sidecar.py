"""`nla_meta.yaml` — the checkpoint contract. Every NLA constant comes from here.

Never hardcode injection token IDs, prompt templates or scale factors (plan rule).
`verify_*` functions assert the sidecar against the live tokenizer before any
inference, catching tokenizer drift and template drift (the two silent failure
modes documented in the official docs/inference.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NLASidecar:
    role: str  # "av" | "ar"
    d_model: int
    extraction_layer_index: int
    injection_scale: float | None  # AV only; AR sidecars carry null
    mse_scale: float
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    critic_suffix_ids: tuple[int, ...] | None
    av_template: str
    ar_template: str
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, meta: dict[str, Any]) -> NLASidecar:
        if meta.get("kind") != "nla_model":
            raise ValueError(f"not an NLA model sidecar (kind={meta.get('kind')!r})")
        role = meta["role"]
        if role not in ("av", "ar"):
            raise ValueError(f"unknown sidecar role {role!r}")
        ext = meta.get("extraction", {}) or {}
        tok = meta["tokens"]
        tmpl = meta["prompt_templates"]
        layer = meta.get("extraction_layer_index")
        if layer is None:
            layer = (meta.get("critic") or {}).get("extraction_layer_index")
        if layer is None:
            raise ValueError("sidecar has no extraction_layer_index")
        if ext.get("mse_scale") is None:
            raise ValueError("sidecar mse_scale is null (raw-MSE checkpoints are unsupported)")
        inj_scale = ext.get("injection_scale")
        if role == "av" and inj_scale is None:
            raise ValueError("AV sidecar must define extraction.injection_scale")
        suffix = tok.get("critic_suffix_ids")
        return cls(
            role=role,
            d_model=int(meta["d_model"]),
            extraction_layer_index=int(layer),
            injection_scale=None if inj_scale is None else float(inj_scale),
            mse_scale=float(ext["mse_scale"]),
            injection_char=tok["injection_char"],
            injection_token_id=int(tok["injection_token_id"]),
            injection_left_neighbor_id=int(tok["injection_left_neighbor_id"]),
            injection_right_neighbor_id=int(tok["injection_right_neighbor_id"]),
            critic_suffix_ids=None if suffix is None else tuple(int(x) for x in suffix),
            av_template=tmpl.get("av") or tmpl["actor"],
            ar_template=tmpl.get("ar") or tmpl["critic"],
            raw=meta,
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> NLASidecar:
        path = Path(checkpoint_dir) / "nla_meta.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no nla_meta.yaml in {checkpoint_dir!s}")
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))

    # -- prompts -------------------------------------------------------------
    def av_user_content(self) -> str:
        return self.av_template.format(injection_char=self.injection_char)

    def ar_prompt(self, explanation: str) -> str:
        return self.ar_template.format(explanation=explanation)


def av_prompt_ids(sidecar: NLASidecar, tokenizer: Any) -> list[int]:
    """Canonical AV prompt token IDs (one-step chat-template tokenization, as in training)."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sidecar.av_user_content()}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "keys"):  # some transformers versions return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def injection_position(sidecar: NLASidecar, ids: list[int]) -> int:
    """The unique position holding the injection token with the sidecar's neighbours."""
    hits = [
        p
        for p, t in enumerate(ids)
        if t == sidecar.injection_token_id
        and 0 < p < len(ids) - 1
        and ids[p - 1] == sidecar.injection_left_neighbor_id
        and ids[p + 1] == sidecar.injection_right_neighbor_id
    ]
    if len(hits) != 1:
        raise AssertionError(
            f"expected exactly 1 neighbour-checked injection site, found {len(hits)}; "
            "template drift or tokenizer mismatch"
        )
    return hits[0]


def verify_av_tokenizer(sidecar: NLASidecar, tokenizer: Any) -> int:
    """Assert sidecar ↔ tokenizer agreement. Returns the injection position."""
    live = tokenizer.encode(sidecar.injection_char, add_special_tokens=False)
    if live != [sidecar.injection_token_id]:
        raise AssertionError(
            f"tokenizer drift: {sidecar.injection_char!r} -> {live}, "
            f"sidecar says [{sidecar.injection_token_id}]"
        )
    ids = av_prompt_ids(sidecar, tokenizer)
    if sum(1 for t in ids if t == sidecar.injection_token_id) != 1:
        raise AssertionError("injection token must appear exactly once in the AV prompt")
    return injection_position(sidecar, ids)


def verify_ar_tokenizer(sidecar: NLASidecar, tokenizer: Any) -> None:
    """The AR reads the LAST token; check the template suffix tokenizes as in training."""
    if sidecar.critic_suffix_ids is None:
        raise AssertionError("AR sidecar must define tokens.critic_suffix_ids")
    ids = tokenizer(sidecar.ar_prompt("probe explanation"), add_special_tokens=True)["input_ids"]
    n = len(sidecar.critic_suffix_ids)
    if tuple(ids[-n:]) != sidecar.critic_suffix_ids:
        raise AssertionError(
            f"AR suffix drift: tokens {ids[-n:]} != sidecar {list(sidecar.critic_suffix_ids)}"
        )
    bos = tokenizer.bos_token_id
    if bos is not None and ids[0] != bos:
        raise AssertionError("AR was trained with a BOS prefix (add_special_tokens=True)")
