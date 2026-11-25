"""Residual-stream activation extraction, matching the official stage-0 conventions.

* "layer K" = output of decoder block K (post-residual) = HF `hidden_states[K+1]`.
  We read it with a forward hook on `model.model.layers[K]` (as the official
  HFExtractor does) so a truncated (K+1)-block model gives the same vector without
  the final norm.
* Positions: index >= 50 and not a special token, sampled with a per-document RNG
  keyed on sha256(f"{seed}|{doc_id}") (official `_sample_positions`).
* Raw text → `tokenizer(text, add_special_tokens=True)`; chat corpora → the model's
  chat template. Max length 4096.
* Never extract from 4/8-bit weights; dtype + GPU are recorded with every file.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

MIN_POSITION = 50  # official stage0_extract._MIN_POSITION
MAX_LENGTH = 4096


def sample_positions(
    token_ids: Sequence[int],
    n_positions: int,
    special_ids: set[int],
    doc_id: str,
    seed: int,
    min_position: int = MIN_POSITION,
) -> list[int]:
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [i for i, t in enumerate(token_ids) if i >= min_position and t not in special_ids]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(n_positions, len(candidates)))


def chat_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)[:MAX_LENGTH]


def text_token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)[
        "input_ids"
    ]


def _decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(f"{type(model).__name__} has no .model.layers")
    return layers


def assert_not_quantized(model: Any) -> None:
    """Plan rule: activations only from unquantized (fp16/bf16/fp32) weights."""
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise AssertionError("refusing to extract activations from a 4/8-bit quantized model")
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is not None:
        raise AssertionError(f"refusing to extract from quantized model ({qc})")
    for p in model.parameters():
        if p.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise AssertionError(f"unexpected parameter dtype {p.dtype}")


@dataclass
class Extraction:
    hidden: torch.Tensor  # [seq, d] float32 CPU
    token_ids: list[int]


class LayerExtractor:
    """Captures block-K output for a batch of token-id sequences (right-padded)."""

    def __init__(self, model: Any, layer_index: int, pad_token_id: int) -> None:
        assert_not_quantized(model)
        self.model = model.eval()
        self.layer_index = layer_index
        self.pad_token_id = pad_token_id
        layers = _decoder_layers(model)
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer {layer_index} out of range ({len(layers)} blocks)")
        self._layer = layers[layer_index]

    @torch.no_grad()
    def extract(self, batch_ids: Sequence[Sequence[int]]) -> list[Extraction]:
        captured: dict[str, torch.Tensor] = {}

        def hook(_m: Any, _i: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach().clone()

        lengths = [len(x) for x in batch_ids]
        t = max(lengths)
        ids = torch.full((len(batch_ids), t), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(list(x), dtype=torch.long)
            mask[i, : len(x)] = 1
        device = self.model.get_input_embeddings().weight.device
        handle = self._layer.register_forward_hook(hook)
        try:
            self.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
        finally:
            handle.remove()
        h = captured["h"]
        if not torch.isfinite(h).all():
            raise FloatingPointError("non-finite activations (fp16 overflow?)")
        h = h.float().cpu()
        return [
            Extraction(hidden=h[i, :n].clone(), token_ids=list(batch_ids[i]))
            for i, n in enumerate(lengths)
        ]


class ActivationMonitor:
    """fp16 validity check: hook EVERY decoder block, track max |h| and non-finite outputs.

    Use as a context manager around forward/generate calls. `report()` gives per-block
    max-abs (fp16 overflows at 65504) and the count of forwards with inf/NaN.
    """

    def __init__(self, model: Any) -> None:
        self.layers = list(_decoder_layers(model))
        self.max_abs = [0.0] * len(self.layers)
        self.nonfinite = [0] * len(self.layers)
        self._handles: list[Any] = []

    def _hook(self, i: int) -> Any:
        def fn(_m: Any, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            finite = torch.isfinite(h)
            if not bool(finite.all()):
                self.nonfinite[i] += 1
            m = h.detach().abs().masked_fill(~finite, 0).amax()
            self.max_abs[i] = max(self.max_abs[i], float(m))

        return fn

    def __enter__(self) -> ActivationMonitor:
        self._handles = [
            layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def report(self) -> dict[str, Any]:
        return {
            "max_abs_per_block": [round(x, 3) for x in self.max_abs],
            "max_abs": max(self.max_abs, default=0.0),
            "nonfinite_forwards_per_block": list(self.nonfinite),
            "any_nonfinite": any(self.nonfinite),
        }


def load_truncated_model(
    model_name: str, layer_index: int, dtype: torch.dtype, device_map: str | dict = "auto"
) -> Any:
    """Load only blocks 0..K (K+1 blocks): ~10 GiB for Qwen2.5-7B L20 in fp16/bf16."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_hidden_layers = layer_index + 1
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=cfg, dtype=dtype, device_map=device_map
    )
    assert_not_quantized(model)
    return model.eval()
