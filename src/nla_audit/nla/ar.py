"""Activation Reconstructor (AR): explanation text → predicted activation.

Truncated (K+1)-block LM, final norm → Identity, lm_head stripped, bias-free
Linear(d, d) `value_head` loaded from value_head.safetensors; read at the last token.
Prompts are the sidecar's raw AR template tokenized with add_special_tokens=True.
Batches are right-padded and read at each row's own last real token.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nla_audit.nla.metrics import cosine, mse_nrm
from nla_audit.nla.sidecar import NLASidecar, verify_ar_tokenizer

_FINAL_NORM_ATTRS = ("norm", "final_layernorm", "ln_f")


class ARScorer:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        dtype: torch.dtype,
        device_map: str | dict = "auto",
        tokenizer: Any | None = None,
    ) -> None:
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = Path(checkpoint_dir)
        self.sidecar = NLASidecar.load(checkpoint_dir)
        if self.sidecar.role != "ar":
            raise ValueError("ARScorer needs the AR sidecar")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(str(checkpoint_dir))
        verify_ar_tokenizer(self.sidecar, self.tokenizer)

        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), dtype=dtype, device_map=device_map
        )
        n_layers = backbone.config.num_hidden_layers
        expected = self.sidecar.extraction_layer_index + 1
        if n_layers != expected:
            raise AssertionError(f"AR has {n_layers} blocks; expected K+1 = {expected}")
        backbone.lm_head = torch.nn.Identity()
        inner = backbone.model
        for attr in _FINAL_NORM_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(f"no final norm attribute on {type(inner).__name__}")
        self.backbone = backbone.eval()
        d = backbone.config.hidden_size
        if d != self.sidecar.d_model:
            raise AssertionError(f"AR d={d} != sidecar d_model={self.sidecar.d_model}")
        head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head.load_state_dict(load_file(str(checkpoint_dir / "value_head.safetensors")))
        out_device = self._last_block_device()
        self.value_head = head.to(out_device).eval()

    def _last_block_device(self) -> torch.device:
        return next(self.backbone.model.layers[-1].parameters()).device

    @property
    def input_device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    @torch.no_grad()
    def reconstruct(self, explanations: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Raw (un-normalised) predicted vectors, float32 [N, d]."""
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        outs: list[np.ndarray] = []
        for start in range(0, len(explanations), batch_size):
            chunk = explanations[start : start + batch_size]
            seqs = [
                self.tokenizer(self.sidecar.ar_prompt(e), add_special_tokens=True)["input_ids"]
                for e in chunk
            ]
            t = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), t), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            h = self.backbone.model(
                input_ids=ids.to(self.input_device),
                attention_mask=mask.to(self.input_device),
                use_cache=False,
            ).last_hidden_state
            last = torch.tensor([len(s) - 1 for s in seqs], device=h.device)
            h_last = h[torch.arange(len(seqs), device=h.device), last]
            pred = self.value_head(h_last.to(self.value_head.weight.device))
            outs.append(pred.float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def score(
        self, explanations: Sequence[str], gold: np.ndarray, batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pred, per-row mse_nrm, per-row cosine)."""
        pred = self.reconstruct(explanations, batch_size=batch_size)
        return pred, mse_nrm(gold, pred, self.sidecar.mse_scale), cosine(gold, pred)
