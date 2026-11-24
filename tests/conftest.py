from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

FIXTURES = Path(__file__).parent / "fixtures"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="session")
def av_meta() -> dict:
    return yaml.safe_load((FIXTURES / "qwen7b_L20_av_nla_meta.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ar_meta() -> dict:
    return yaml.safe_load((FIXTURES / "qwen7b_L20_ar_nla_meta.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """The real Qwen2.5 tokenizer (a few MB; needs internet the first time)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except Exception as e:  # offline
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


TINY_LAYERS = 4


def make_tiny_qwen(vocab_size: int, n_layers: int = TINY_LAYERS, seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tiny_qwen(qwen_tokenizer):
    return make_tiny_qwen(len(qwen_tokenizer))


def tiny_meta(meta: dict, *, d_model: int = 64, layer: int = 2) -> dict:
    """Real sidecar with d_model/layer swapped for the tiny model (tokens/templates unchanged)."""
    m = copy.deepcopy(meta)
    m["d_model"] = d_model
    m["extraction_layer_index"] = layer
    if "critic" in m:
        m["critic"]["extraction_layer_index"] = layer
    m["extraction"]["mse_scale"] = d_model**0.5
    return m
