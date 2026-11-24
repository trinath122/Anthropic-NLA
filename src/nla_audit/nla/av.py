"""Activation Verbalizer (AV): inject a vector as one token embedding, sample an explanation.

Batched HF `generate(inputs_embeds=...)`. The AV prompt is identical for every
activation, so batches need no padding. Sampling is pure temperature sampling
(top_p=1, top_k=0, no repetition penalty), overriding Qwen's generation_config
defaults, to match the paper's "sample at T=1".
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nla_audit.nla.sidecar import NLASidecar, av_prompt_ids, verify_av_tokenizer

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
_CJK_RE = re.compile(r"[　-〿぀-ヿ㈀-㋿㐀-䶿一-鿿]")
# model_types whose embedding forward multiplies by sqrt(d) (official arch_adapters list)
_SCALED_EMBED_MODEL_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text", "t5"})


@dataclass(frozen=True)
class Decode:
    raw: str
    explanation: str | None  # None when the <explanation> tags are missing (truncated)

    @property
    def parsed(self) -> bool:
        return self.explanation is not None

    @property
    def text(self) -> str:
        return self.explanation if self.explanation is not None else self.raw


def parse_explanation(raw: str) -> Decode:
    m = EXPLANATION_RE.search(raw)
    return Decode(raw=raw, explanation=None if m is None else m.group(1).strip())


def cjk_fraction(text: str) -> float:
    """Fraction of non-space chars that are CJK — the injection-failure smell."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _CJK_RE.match(c)) / len(chars)


def embed_scale_for(model: Any) -> float:
    cfg = getattr(model.config, "text_config", model.config)
    if getattr(cfg, "model_type", "") in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(cfg.hidden_size)
    return 1.0


class AVRunner:
    def __init__(self, model: Any, tokenizer: Any, sidecar: NLASidecar) -> None:
        if sidecar.role != "av":
            raise ValueError("AVRunner needs the AV sidecar")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.sidecar = sidecar
        self.inj_pos = verify_av_tokenizer(sidecar, tokenizer)
        self.prompt_ids = av_prompt_ids(sidecar, tokenizer)
        hidden = model.get_input_embeddings().weight.shape[1]
        if hidden != sidecar.d_model:
            raise AssertionError(f"model d={hidden} != sidecar d_model={sidecar.d_model}")
        self.embed_scale = embed_scale_for(model)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.get_input_embeddings().weight.dtype

    def build_embeds(self, vectors: torch.Tensor) -> torch.Tensor:
        """[N, d] raw activations → [N, T, d] prompt embeddings with the vector injected."""
        if vectors.ndim != 2 or vectors.shape[1] != self.sidecar.d_model:
            raise ValueError(f"expected [N, {self.sidecar.d_model}], got {tuple(vectors.shape)}")
        v = vectors.to(self.device, torch.float32)
        if not torch.isfinite(v).all():
            raise ValueError("activation contains NaN/Inf")
        assert self.sidecar.injection_scale is not None
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.sidecar.injection_scale
        ids = torch.tensor([self.prompt_ids], device=self.device)
        base = self.model.get_input_embeddings()(ids).float() * self.embed_scale  # [1, T, d]
        out = base.expand(v.shape[0], -1, -1).clone()
        out[:, self.inj_pos] = v
        return out.to(self.dtype)

    @torch.no_grad()
    def generate(
        self,
        vectors: np.ndarray | torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int = 0,
        batch_size: int = 32,
    ) -> list[Decode]:
        vecs = torch.as_tensor(np.asarray(vectors, dtype=np.float32))
        results: list[Decode] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        for b, start in enumerate(range(0, vecs.shape[0], batch_size)):
            chunk = vecs[start : start + batch_size]
            emb = self.build_embeds(chunk)
            mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
            torch.manual_seed(seed * 1_000_003 + b)
            out = self.model.generate(
                inputs_embeds=emb,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=pad_id,
            )
            # With inputs_embeds only, HF returns just the generated tokens.
            texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend(parse_explanation(t) for t in texts)
        return results


def generate_samples(
    runner: AVRunner,
    vectors: np.ndarray,
    n_samples: int,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    batch_size: int,
) -> list[list[Decode]]:
    """K independent samples per vector: result[i][k]."""
    per_sample: Sequence[list[Decode]] = [
        runner.generate(
            vectors,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed * 100 + k,
            batch_size=batch_size,
        )
        for k in range(n_samples)
    ]
    return [[per_sample[k][i] for k in range(n_samples)] for i in range(len(vectors))]
