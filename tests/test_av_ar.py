"""AV/AR plumbing on a tiny random Qwen with the REAL Qwen tokenizer and sidecar templates."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from nla_audit.nla.ar import ARScorer
from nla_audit.nla.av import AVRunner, cjk_fraction, parse_explanation
from nla_audit.nla.sidecar import NLASidecar

from .conftest import make_tiny_qwen, tiny_meta

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def av(av_meta, qwen_tokenizer, tiny_qwen):
    return AVRunner(tiny_qwen, qwen_tokenizer, NLASidecar.from_dict(tiny_meta(av_meta)))


def test_injection_row_has_injection_scale_norm(av):
    v = torch.randn(3, 64) * 7
    emb = av.build_embeds(v).float()
    norms = emb[:, av.inj_pos].norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.full((3,), av.sidecar.injection_scale), rtol=1e-3, atol=0.2
    )
    # every other position is the plain token embedding
    plain = av.model.get_input_embeddings()(torch.tensor([av.prompt_ids])).float()[0]
    mask = torch.ones(len(av.prompt_ids), dtype=torch.bool)
    mask[av.inj_pos] = False
    torch.testing.assert_close(emb[0][mask], plain[mask])


def test_rejects_bad_vectors(av):
    with pytest.raises(ValueError):
        av.build_embeds(torch.randn(2, 65))
    bad = torch.randn(1, 64)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        av.build_embeds(bad)


def test_generate_is_seeded(av):
    vecs = np.random.default_rng(0).normal(size=(3, 64)).astype(np.float32)
    a = av.generate(vecs, max_new_tokens=8, seed=5, batch_size=2)
    b = av.generate(vecs, max_new_tokens=8, seed=5, batch_size=2)
    assert len(a) == 3
    assert [d.raw for d in a] == [d.raw for d in b]


def test_parse_and_cjk():
    d = parse_explanation("x <explanation>  The text is about rivers. </explanation> y")
    assert d.parsed and d.explanation == "The text is about rivers."
    assert not parse_explanation("<explanation> truncated...").parsed
    assert cjk_fraction("hello") == 0.0
    assert cjk_fraction("中文中文") == 1.0


@pytest.fixture(scope="module")
def ar(tmp_path_factory, ar_meta, qwen_tokenizer):
    layer = 2
    d = tmp_path_factory.mktemp("tiny_ar")
    model = make_tiny_qwen(len(qwen_tokenizer), n_layers=layer + 1, seed=3)
    model.save_pretrained(d)
    qwen_tokenizer.save_pretrained(d)
    torch.manual_seed(0)
    save_file({"weight": torch.randn(64, 64) * 0.1}, str(d / "value_head.safetensors"))
    (d / "nla_meta.yaml").write_text(
        yaml.safe_dump(tiny_meta(ar_meta, layer=layer), allow_unicode=True), encoding="utf-8"
    )
    return ARScorer(d, dtype=torch.float32, device_map="cpu")


def test_ar_batching_matches_single(ar):
    texts = ["short one", "a considerably longer explanation about rivers and boats and tides"]
    batched = ar.reconstruct(texts, batch_size=2)
    single = np.stack([ar.reconstruct([t])[0] for t in texts])
    np.testing.assert_allclose(batched, single, rtol=1e-4, atol=1e-5)


def test_ar_score_consistency(ar):
    gold = np.random.default_rng(1).normal(size=(2, 64))
    _, mse, cos = ar.score(["alpha", "beta"], gold)
    np.testing.assert_allclose(mse, 2 * (1 - cos), atol=1e-6)


def test_ar_rejects_wrong_depth(tmp_path, ar_meta, qwen_tokenizer):
    model = make_tiny_qwen(len(qwen_tokenizer), n_layers=2)
    model.save_pretrained(tmp_path)
    qwen_tokenizer.save_pretrained(tmp_path)
    save_file({"weight": torch.zeros(64, 64)}, str(tmp_path / "value_head.safetensors"))
    (tmp_path / "nla_meta.yaml").write_text(
        yaml.safe_dump(tiny_meta(ar_meta, layer=2), allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="blocks"):
        ARScorer(tmp_path, dtype=torch.float32, device_map="cpu")
