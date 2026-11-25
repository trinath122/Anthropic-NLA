"""Step 1 pipeline: report math, data filters, and an end-to-end run on tiny models with resume."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from nla_audit.config import load_config
from nla_audit.data.corpora import Doc, shuffled_pool, ufw_docs, wildchat_docs
from nla_audit.experiments import step1_baseline as s1
from nla_audit.nla.extract import ActivationMonitor
from nla_audit.runtime.resume import Checkpointer

from .conftest import make_tiny_qwen, tiny_meta

STEP1_CFG = load_config(__import__("pathlib").Path(__file__).parents[1] / "configs" / "step1.yaml")
RCFG = STEP1_CFG["report"]


# --------------------------------------------------------------------------- pure functions
def test_shuffled_permutation_has_no_fixed_points():
    for n in (2, 3, 17, 1000):
        perm = s1.shuffled_permutation(n, seed=1)
        assert sorted(perm) == list(range(n))
        assert not np.any(perm == np.arange(n))


def test_report_perfect_and_shuffled():
    rng = np.random.default_rng(0)
    gold = rng.normal(size=(200, 16)) + 3.0  # shared mean direction, like real activations
    sources = np.array(["ufw", "wildchat"] * 100)
    decodes = [{"raw": "<explanation>x</explanation>", "explanation": "x"}] * 200
    rep = s1.compute_report(gold, gold.copy(), decodes, sources, mse_scale=4.0, rcfg=RCFG, seed=0)
    assert rep["overall"]["fve_own"]["fve"] == pytest.approx(1.0)
    assert rep["shuffled_control"]["fve_own"] < 0  # mismatched pairs are worse than the mean
    assert rep["overall"]["parse_fail_rate"] == 0.0
    assert rep["verdict"]["in_band"] is False  # 1.0 is above the 0.60-0.78 band
    assert rep["ufw"]["n"] == rep["wildchat"]["n"] == 100


def test_report_mean_predictor_is_zero_and_fails():
    rng = np.random.default_rng(1)
    gold = rng.normal(size=(300, 16)) + 3.0
    g = gold / np.linalg.norm(gold, axis=1, keepdims=True) * 4.0
    pred = np.repeat(g.mean(axis=0, keepdims=True), 300, axis=0)
    decodes = [{"raw": "中文", "explanation": None}] * 300
    rep = s1.compute_report(
        gold, pred, decodes, np.array(["ufw"] * 300), mse_scale=4.0, rcfg=RCFG, seed=0
    )
    # predicting the (normalised-then-averaged) mean direction is close to FVE 0, not exactly
    assert abs(rep["overall"]["fve_own"]["fve"]) < 0.1
    assert rep["overall"]["parse_fail_rate"] == 1.0
    assert rep["overall"]["cjk_any_rate"] == 1.0
    assert rep["verdict"]["pass"] is False


def test_compare_tags_identical_passes_and_noise_fails():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(100, 32)) + 2
    p = g + rng.normal(size=g.shape)
    same = s1.compare_tags(g, g, p, p, mse_scale=1.0, cos_min=0.999, n_boot=200, seed=0)
    assert same["pass"] and same["activation_cos"]["min"] == pytest.approx(1.0)
    noisy = s1.compare_tags(
        g, g + rng.normal(size=g.shape), p, p, mse_scale=1.0, cos_min=0.999, n_boot=200, seed=0
    )
    assert not noisy["pass"]


def test_token_batches_respect_limits():
    seqs = [[0] * n for n in (10, 10, 50, 5, 5, 5, 200, 3)]
    batches = s1.token_batches(seqs, max_n=3, max_tokens=60)
    assert [x for b in batches for x in b] == seqs  # order preserved, nothing dropped
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or len(b) * max(map(len, b)) <= 60
    assert [len(b) for b in batches] == [2, 1, 3, 1, 1]


def test_extractor_works_on_decoder_body_without_lm_head():
    from transformers import Qwen2Model

    full = make_tiny_qwen(100, n_layers=3)
    body = Qwen2Model(full.config)
    body.load_state_dict(full.model.state_dict())
    from nla_audit.nla.extract import LayerExtractor

    ids = [list(range(3, 63))]
    a = LayerExtractor(full, 1, 0).extract(ids)[0].hidden
    b = LayerExtractor(body.eval(), 1, 0).extract(ids)[0].hidden
    torch.testing.assert_close(a, b)


def test_monitor_flags_nonfinite():
    model = make_tiny_qwen(100, n_layers=2)
    with ActivationMonitor(model) as mon:
        model(torch.tensor([[1, 2, 3]]))
    assert not mon.report()["any_nonfinite"]
    model.model.layers[1].mlp.down_proj.weight.data[0, 0] = float("inf")
    with ActivationMonitor(model) as mon:
        model(torch.tensor([[1, 2, 3]]))
    r = mon.report()
    assert r["any_nonfinite"] and r["nonfinite_forwards_per_block"] == [0, 1]
    assert len(model.model.layers[0]._forward_hooks) == 0  # hooks removed


def test_corpus_filters():
    rows = [
        {"content": "short"},
        {"content": "x" * 500},
        {"content": "x" * 500},  # duplicate
    ]
    docs = list(ufw_docs(rows, min_chars=400))
    assert len(docs) == 2 and docs[0].doc_id == docs[1].doc_id
    assert len(shuffled_pool(docs, pool=10, seed=0)) == 1  # deduplicated

    conv = [
        {"role": "user", "content": "hi", "turn_identifier": 1},
        {"role": "assistant", "content": "yo"},
    ]
    wrows = [
        {
            "conversation": conv,
            "language": "English",
            "toxic": False,
            "redacted": False,
            "conversation_hash": "a",
        },
        {
            "conversation": conv,
            "language": "English",
            "toxic": True,
            "redacted": False,
            "conversation_hash": "b",
        },
        {
            "conversation": conv,
            "language": "English",
            "toxic": False,
            "redacted": True,
            "conversation_hash": "c",
        },
        {
            "conversation": conv,
            "language": "Chinese",
            "toxic": False,
            "redacted": False,
            "conversation_hash": "d",
        },
    ]
    w = list(wildchat_docs(wrows))
    assert len(w) == 1
    assert w[0].messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_shuffled_pool_is_deterministic():
    docs = [Doc(f"d{i}", "ufw", "text", text="t") for i in range(50)]
    a = [d.doc_id for d in shuffled_pool(docs, 30, seed=3)]
    b = [d.doc_id for d in shuffled_pool(docs, 30, seed=3)]
    assert a == b and len(a) == 30 and a != [f"d{i}" for i in range(30)]


# --------------------------------------------------------------------------- end-to-end (tiny)
LAYER = 2


def _tiny_cfg() -> dict:
    cfg = copy.deepcopy(STEP1_CFG)
    cfg["layer"] = LAYER
    cfg["shard_size"] = 4
    cfg["data"]["ufw"].update(n=6, pool=12)
    cfg["data"]["wildchat"].update(n=6, pool=12)
    cfg["av"]["max_new_tokens"] = 6
    cfg["report"]["n_boot"] = 50
    cfg["profiles"]["tiny"] = {
        "extract_dtypes": ["lo", "hi"],  # two tags exercise the precision check
        "nla_dtype": "fp32",
        "extract_batch": 3,
        "av_batch": 4,
        "ar_batch": 4,
    }
    return cfg


def _streams():
    words = [
        *("the", "river", "runs", "past", "old", "mills", "and"),
        *("quiet", "towns", "toward", "the", "grey", "sea"),
    ]
    ufw = [
        Doc(
            f"ufw-{i}", "ufw", "text", text=" ".join(words[(i + j) % len(words)] for j in range(90))
        )
        for i in range(15)
    ]
    chats = [
        Doc(
            f"wc-{i}",
            "wildchat",
            "chat",
            messages=[
                {"role": "user", "content": " ".join(words * 3) + f" question {i}"},
                {"role": "assistant", "content": " ".join(reversed(words * 3))},
            ],
        )
        for i in range(15)
    ]
    return {"ufw": ufw, "wildchat": chats}


@pytest.mark.network
def test_end_to_end_tiny_with_resume(tmp_path, qwen_tokenizer, av_meta, ar_meta):
    from nla_audit.nla.ar import ARScorer
    from nla_audit.nla.av import AVRunner
    from nla_audit.nla.sidecar import NLASidecar

    vocab = len(qwen_tokenizer)
    base = make_tiny_qwen(vocab, n_layers=LAYER + 1, seed=0)
    av_model = make_tiny_qwen(vocab, n_layers=4, seed=1)
    ar_dir = tmp_path / "ar"
    make_tiny_qwen(vocab, n_layers=LAYER + 1, seed=2).save_pretrained(ar_dir)
    qwen_tokenizer.save_pretrained(ar_dir)
    torch.manual_seed(0)
    save_file({"weight": torch.randn(64, 64) * 0.1}, str(ar_dir / "value_head.safetensors"))
    (ar_dir / "nla_meta.yaml").write_text(
        yaml.safe_dump(tiny_meta(ar_meta, layer=LAYER), allow_unicode=True), encoding="utf-8"
    )

    calls = {"extract": 0, "av": 0, "ar": 0}
    fail_ar = {"on": True}

    def load_extractor(tag):
        calls["extract"] += 1
        return base

    def load_av():
        calls["av"] += 1
        return AVRunner(
            av_model, qwen_tokenizer, NLASidecar.from_dict(tiny_meta(av_meta, layer=LAYER))
        )

    def load_ar():
        calls["ar"] += 1
        if fail_ar["on"]:
            raise RuntimeError("simulated session death before the AR stage")
        return ARScorer(ar_dir, dtype=torch.float32, device_map="cpu")

    be = s1.Backends(qwen_tokenizer, _streams, load_extractor, load_av, load_ar)
    cfg = _tiny_cfg()
    root = tmp_path / "runs"

    # session 1 dies at the AR stage
    with pytest.raises(RuntimeError, match="simulated"):
        s1.run(cfg, "tiny", Checkpointer("step1/tiny", root), be)
    assert calls == {"extract": 2, "av": 1, "ar": 1}

    # session 2 resumes: no re-extraction, no re-generation
    fail_ar["on"] = False
    summary = s1.run(cfg, "tiny", Checkpointer("step1/tiny", root), be)
    assert calls == {"extract": 2, "av": 1, "ar": 2}

    run_dir = root / "step1" / "tiny"
    items = __import__("pandas").read_parquet(run_dir / "items.parquet")
    assert len(items) == 12 and (items["position"] >= 50).all()
    assert set(items["source"]) == {"ufw", "wildchat"}
    for name in ("summary.json", "report-lo.json", "results-hi.parquet", "l2_mean_norm.json"):
        assert (run_dir / name).exists(), name
    rep = summary["tags"]["lo"]
    assert rep["n"] == 12 and np.isfinite(rep["overall"]["fve_own"]["fve"])
    assert summary["precision_check"]["activation_cos"]["min"] == pytest.approx(1.0)
    assert summary["mean_norm"]["n"] == 12 and summary["mean_norm"]["mean_norm"] > 0
    assert json.loads((run_dir / "monitor-extract-lo.json").read_text())["any_nonfinite"] is False

    # activation at position p equals the full-sequence activation (prefix cut is exact)
    from nla_audit.nla.extract import LayerExtractor, chat_token_ids

    row = items[items.source == "wildchat"].iloc[0]
    doc = next(d for d in _streams()["wildchat"] if d.doc_id == row.doc_id)
    full = LayerExtractor(base, LAYER, 0).extract([chat_token_ids(qwen_tokenizer, doc.messages)])
    acts = s1.load_acts(Checkpointer("step1/tiny", root), "lo", 12, 4)
    i = int(items.index[items.doc_id == row.doc_id][0])
    np.testing.assert_allclose(acts[i], full[0].hidden[row.position].numpy(), rtol=1e-4, atol=1e-5)

    # a third run is a no-op
    s1.run(cfg, "tiny", Checkpointer("step1/tiny", root), be)
    assert calls == {"extract": 2, "av": 1, "ar": 2}
