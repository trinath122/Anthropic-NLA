from __future__ import annotations

import pytest
import torch

from nla_audit.nla.extract import (
    MIN_POSITION,
    LayerExtractor,
    assert_not_quantized,
    sample_positions,
)

from .conftest import TINY_LAYERS, make_tiny_qwen

VOCAB = 1000


def test_positions_deterministic_and_valid():
    ids = list(range(200))
    specials = {60, 61}
    a = sample_positions(ids, 10, specials, "doc-1", seed=42)
    b = sample_positions(ids, 10, specials, "doc-1", seed=42)
    c = sample_positions(ids, 10, specials, "doc-2", seed=42)
    assert a == b
    assert a != c
    assert all(p >= MIN_POSITION for p in a)
    assert not set(a) & specials


def test_short_doc_has_no_positions():
    assert sample_positions(list(range(40)), 5, set(), "d", 0) == []


@pytest.mark.parametrize("layer", [0, 2, TINY_LAYERS - 1])
def test_hook_equals_hidden_states_k_plus_1(layer):
    """'Layer K' = output of block K = HF hidden_states[K+1] (official convention)."""
    model = make_tiny_qwen(VOCAB)
    ids = [list(range(5, 75))]
    got = LayerExtractor(model, layer, pad_token_id=0).extract(ids)[0].hidden
    with torch.no_grad():
        hs = model(torch.tensor(ids), output_hidden_states=True).hidden_states
    if layer == TINY_LAYERS - 1:
        # HF applies the final norm to the LAST entry of hidden_states; the hook does not.
        assert not torch.allclose(got, hs[layer + 1][0].float(), atol=1e-5)
    else:
        torch.testing.assert_close(got, hs[layer + 1][0].float())
        assert not torch.allclose(got, hs[layer][0].float())  # the README's off-by-one


def test_truncated_model_matches_full():
    full = make_tiny_qwen(VOCAB, n_layers=TINY_LAYERS, seed=1)
    trunc = make_tiny_qwen(VOCAB, n_layers=3, seed=99)
    trunc.load_state_dict(full.state_dict(), strict=False)  # blocks 0..2 + embeddings
    ids = [list(range(10, 90))]
    a = LayerExtractor(full, 2, 0).extract(ids)[0].hidden
    b = LayerExtractor(trunc, 2, 0).extract(ids)[0].hidden
    torch.testing.assert_close(a, b)


def test_right_padding_does_not_change_activations():
    model = make_tiny_qwen(VOCAB)
    ex = LayerExtractor(model, 1, pad_token_id=0)
    short, long = list(range(3, 63)), list(range(7, 107))
    batched = ex.extract([short, long])
    alone = ex.extract([short])[0]
    torch.testing.assert_close(batched[0].hidden, alone.hidden, atol=1e-5, rtol=1e-4)
    assert batched[0].hidden.shape[0] == len(short)


def test_extract_never_runs_lm_head():
    """The LM head's [B, T, vocab] logits OOM'd a 24 GB L4; extraction must skip it."""
    model = make_tiny_qwen(VOCAB)
    called = []
    model.lm_head.register_forward_hook(lambda *a: called.append(1))
    LayerExtractor(model, 1, pad_token_id=0).extract([list(range(3, 63))])
    assert not called


def test_refuses_quantized():
    model = make_tiny_qwen(VOCAB)
    model.is_loaded_in_4bit = True
    with pytest.raises(AssertionError, match="quantized"):
        assert_not_quantized(model)
