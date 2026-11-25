"""Step 1 — baseline FVE of the released Qwen2.5-7B L20 NLA.

Stages (each sharded + resumable through `Checkpointer`; re-running skips finished shards):

1. items     pick 500 UFW + 500 WildChat docs, one position >= 50 each (non-special token).
2. extract   L20 activations for every dtype in the profile (Kaggle: fp16 AND fp32; Colab: bf16),
             with an every-block overflow monitor. Sequences are cut at position+1 (causal).
3. av        one AV explanation per activation (T=1, 150 new tokens).
4. ar        AR reconstruction of each explanation.
5. report    FVE (own μ and the 0.7335 training constant) per source + overall with bootstrap
             CIs, shuffled control, parse-failure and CJK rates, pass/fail vs the
             pre-registered band; fp16-vs-fp32 agreement; the L20 mean-norm artifact
             (amendment 2) used later for off-layer probes.

    python -m nla_audit.experiments.step1_baseline run --config configs/step1.yaml --profile kaggle
    python -m nla_audit.experiments.step1_baseline compare --a step1_baseline/kaggle:fp16 \
        --b step1_baseline/colab:bf16 --sync
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nla_audit.config import load_config, set_seed
from nla_audit.data.corpora import Doc, shuffled_pool
from nla_audit.nla.av import cjk_fraction
from nla_audit.nla.extract import (
    ActivationMonitor,
    LayerExtractor,
    chat_token_ids,
    sample_positions,
    text_token_ids,
)
from nla_audit.nla.metrics import baseline_variance, bootstrap_ci, cosine, fve, mse_nrm
from nla_audit.runtime.resume import Checkpointer

SOURCES = ("ufw", "wildchat")


# --------------------------------------------------------------------------- backends
@dataclass
class Backends:
    """Everything heavy, injected so tests can run the whole pipeline on tiny models.

    Loaders are called lazily, one model at a time, and released before the next.
    """

    tokenizer: Any  # base-model tokenizer (extraction side)
    doc_streams: Callable[[], dict[str, Iterable[Doc]]]
    load_extractor: Callable[[str], Any]  # dtype name -> truncated base model
    load_av: Callable[[], Any]  # -> AVRunner
    load_ar: Callable[[], Any]  # -> ARScorer
    provenance: Callable[[str | None], dict[str, Any]] = lambda dtype: {"dtype": dtype}


def free_gpu() -> None:
    """Call after `del`-ing every reference to a model, before loading the next one."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- stage 1: items
def build_items(cfg: dict, tokenizer: Any, streams: dict[str, Iterable[Doc]]) -> pd.DataFrame:
    seed = int(cfg["seed"])
    special = set(tokenizer.all_special_ids)
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        dcfg = cfg["data"][source]
        pool = shuffled_pool(streams[source], int(dcfg["pool"]), seed)
        kept = 0
        for d in pool:
            if kept >= int(dcfg["n"]):
                break
            ids = (
                text_token_ids(tokenizer, d.text or "")
                if d.kind == "text"
                else chat_token_ids(tokenizer, d.messages or [])
            )
            pos = sample_positions(ids, 1, special, d.doc_id, seed)
            if not pos:
                continue
            p = pos[0]
            rows.append(
                {
                    "doc_id": d.doc_id,
                    "source": source,
                    "position": p,
                    "n_tokens": len(ids),
                    "token_id": ids[p],
                    "prefix_ids": ids[: p + 1],  # causal: activation depends only on these
                }
            )
            kept += 1
        if kept < int(dcfg["n"]):
            raise ValueError(f"{source}: only {kept} usable docs in pool, need {dcfg['n']}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- helpers
def _shards(n: int, size: int) -> list[tuple[int, int, int]]:
    return [(k, s, min(s + size, n)) for k, s in enumerate(range(0, n, size))]


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def load_acts(ck: Checkpointer, tag: str, n: int, shard_size: int) -> np.ndarray:
    return np.concatenate(
        [
            np.load(ck.shard_path(f"extract-{tag}-{k:03d}", ".npy"))
            for k, _, _ in _shards(n, shard_size)
        ]
    )


def load_decodes(ck: Checkpointer, tag: str, n: int, shard_size: int) -> list[dict]:
    out: list[dict] = []
    for k, _, _ in _shards(n, shard_size):
        p = ck.shard_path(f"av-{tag}-{k:03d}", ".jsonl")
        out.extend(json.loads(x) for x in p.read_text(encoding="utf-8").splitlines())
    return out


def load_preds(ck: Checkpointer, tag: str, n: int, shard_size: int) -> np.ndarray:
    return np.concatenate(
        [np.load(ck.shard_path(f"ar-{tag}-{k:03d}", ".npy")) for k, _, _ in _shards(n, shard_size)]
    )


# --------------------------------------------------------------------------- stages 2-4
def stage_extract(
    cfg: dict, prof: dict, ck: Checkpointer, be: Backends, items: pd.DataFrame
) -> None:
    size = int(cfg["shard_size"])
    shards = _shards(len(items), size)
    for tag in prof["extract_dtypes"]:
        todo = [s for s in shards if not ck.is_done(f"extract-{tag}-{s[0]:03d}")]
        if not todo:
            continue
        model = be.load_extractor(tag)
        pad = be.tokenizer.pad_token_id
        ext = LayerExtractor(model, int(cfg["layer"]), pad if pad is not None else 0)
        bs = int(prof["extract_batch"])
        mon_path = ck.local_dir / f"monitor-extract-{tag}.json"
        prev = json.loads(mon_path.read_text()) if mon_path.exists() else None
        with ActivationMonitor(model) as mon:
            for k, s, e in todo:
                prefixes = list(items["prefix_ids"].iloc[s:e])
                acts = []
                for b in range(0, len(prefixes), bs):
                    batch = [list(map(int, x)) for x in prefixes[b : b + bs]]
                    acts.extend(x.hidden[-1].numpy() for x in ext.extract(batch))
                np.save(ck.shard_path(f"extract-{tag}-{k:03d}", ".npy"), np.stack(acts))
                rep = _merge_monitor(prev, mon.report())
                _write_json(mon_path, {**rep, "provenance": be.provenance(tag)})
                ck.mark_done(f"extract-{tag}-{k:03d}")
        del ext, model, mon
        free_gpu()


def _merge_monitor(prev: dict | None, cur: dict) -> dict:
    """Combine monitor stats across resumed sessions (max of max, sum of counts)."""
    if not prev:
        return cur
    mx = [
        max(a, b) for a, b in zip(prev["max_abs_per_block"], cur["max_abs_per_block"], strict=True)
    ]
    pa, ca = prev["nonfinite_forwards_per_block"], cur["nonfinite_forwards_per_block"]
    nf = [a + b for a, b in zip(pa, ca, strict=True)]
    return {
        "max_abs_per_block": mx,
        "max_abs": max(mx, default=0.0),
        "nonfinite_forwards_per_block": nf,
        "any_nonfinite": any(nf),
    }


def stage_av(cfg: dict, prof: dict, ck: Checkpointer, be: Backends, n: int) -> None:
    size = int(cfg["shard_size"])
    todo = [
        (tag, k, s, e)
        for tag in prof["extract_dtypes"]
        for k, s, e in _shards(n, size)
        if not ck.is_done(f"av-{tag}-{k:03d}")
    ]
    if not todo:
        return
    runner = be.load_av()
    mon_path = ck.local_dir / "monitor-av.json"
    prev = json.loads(mon_path.read_text()) if mon_path.exists() else None
    with ActivationMonitor(runner.model) as mon:
        for tag, k, _s, _e in todo:
            vecs = np.load(ck.shard_path(f"extract-{tag}-{k:03d}", ".npy"))
            decs = runner.generate(
                vecs,
                max_new_tokens=int(cfg["av"]["max_new_tokens"]),
                temperature=float(cfg["av"]["temperature"]),
                seed=int(cfg["seed"]) * 10_000 + k,  # same seed for every dtype tag
                batch_size=int(prof["av_batch"]),
            )
            lines = [
                json.dumps({"raw": d.raw, "explanation": d.explanation}, ensure_ascii=False)
                for d in decs
            ]
            ck.shard_path(f"av-{tag}-{k:03d}", ".jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            rep = _merge_monitor(prev, mon.report())
            _write_json(mon_path, {**rep, "dtype": prof["nla_dtype"]})
            ck.mark_done(f"av-{tag}-{k:03d}")
    del runner, mon
    free_gpu()


def explanation_text(d: dict) -> str:
    return d["explanation"] if d["explanation"] is not None else d["raw"]


def stage_ar(cfg: dict, prof: dict, ck: Checkpointer, be: Backends, n: int) -> None:
    size = int(cfg["shard_size"])
    todo = [
        (tag, k)
        for tag in prof["extract_dtypes"]
        for k, _, _ in _shards(n, size)
        if not ck.is_done(f"ar-{tag}-{k:03d}")
    ]
    if not todo:
        return
    scorer = be.load_ar()
    _write_json(ck.local_dir / "mse_scale.json", {"mse_scale": scorer.sidecar.mse_scale})
    for tag, k in todo:
        p = ck.shard_path(f"av-{tag}-{k:03d}", ".jsonl")
        decs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        pred = scorer.reconstruct(
            [explanation_text(d) for d in decs], batch_size=int(prof["ar_batch"])
        )
        np.save(ck.shard_path(f"ar-{tag}-{k:03d}", ".npy"), pred.astype(np.float32))
        ck.mark_done(f"ar-{tag}-{k:03d}")
    del scorer
    free_gpu()


# --------------------------------------------------------------------------- stage 5: report
def _fve_ci(gold, pred, scale, den, n_boot, seed) -> dict[str, float]:
    def stat(idx: np.ndarray) -> float:
        return fve(gold[idx], pred[idx], scale, den)

    point, lo, hi = bootstrap_ci(len(gold), stat, n_boot=n_boot, seed=seed)
    return {"fve": point, "lo": lo, "hi": hi}


def shuffled_permutation(n: int, seed: int) -> np.ndarray:
    """A permutation with no fixed points (every explanation scored against another vector)."""
    order = np.random.default_rng(seed).permutation(n)
    perm = np.empty(n, dtype=int)
    perm[order] = np.roll(order, 1)
    return perm


def compute_report(
    gold: np.ndarray,
    pred: np.ndarray,
    decodes: list[dict],
    sources: np.ndarray,
    *,
    mse_scale: float,
    rcfg: dict,
    seed: int,
) -> dict[str, Any]:
    n_boot = int(rcfg["n_boot"])
    ref = float(rcfg["reference_denominator"])
    parsed = np.array([d["explanation"] is not None for d in decodes])
    cjk = np.array([cjk_fraction(d["raw"]) for d in decodes])
    out: dict[str, Any] = {"n": len(gold)}
    groups = {"overall": np.ones(len(gold), bool)} | {s: sources == s for s in SOURCES}
    for name, m in groups.items():
        if not m.any():
            continue
        g, p = gold[m], pred[m]
        out[name] = {
            "n": int(m.sum()),
            "own_denominator": baseline_variance(g, mse_scale),
            "fve_own": _fve_ci(g, p, mse_scale, None, n_boot, seed),
            "fve_reference": _fve_ci(g, p, mse_scale, ref, n_boot, seed),
            "mean_cos": float(cosine(g, p).mean()),
            "mean_mse_nrm": float(mse_nrm(g, p, mse_scale).mean()),
            "parse_fail_rate": float(1 - parsed[m].mean()),
            "cjk_any_rate": float((cjk[m] > 0).mean()),
            "cjk_mean_fraction": float(cjk[m].mean()),
        }
    if parsed.any():
        out["overall"]["fve_own_parsed_only"] = fve(gold[parsed], pred[parsed], mse_scale)
    perm = shuffled_permutation(len(gold), seed)
    out["shuffled_control"] = {
        "fve_own": fve(gold[perm], pred, mse_scale),
        "fve_reference": fve(gold[perm], pred, mse_scale, ref),
    }
    key = "fve_own" if rcfg["pass_denominator"] == "own" else "fve_reference"
    lo, hi = rcfg["pass_band"]
    point = out["overall"][key]["fve"]
    shuf = out["shuffled_control"][key]
    out["verdict"] = {
        "metric": key,
        "fve": point,
        "in_band": bool(lo <= point <= hi),
        "shuffled_ok": bool(shuf <= float(rcfg["shuffled_max"])),
        "pass": bool(lo <= point <= hi and shuf <= float(rcfg["shuffled_max"])),
        "band": [lo, hi],
        "note": (
            "debug injection only if FVE < band low or shuffled > max (amendment 6); "
            "FVE above band is reported, not debugged"
        ),
    }
    return out


def compare_tags(
    gold_a: np.ndarray,
    gold_b: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    mse_scale: float,
    cos_min: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Precision agreement: per-activation cos(a, b) and paired-bootstrap ΔFVE (own μ)."""
    cos = cosine(gold_a, gold_b)

    def diff(idx: np.ndarray) -> float:
        return fve(gold_a[idx], pred_a[idx], mse_scale) - fve(gold_b[idx], pred_b[idx], mse_scale)

    d, lo, hi = bootstrap_ci(len(gold_a), diff, n_boot=n_boot, seed=seed)
    return {
        "activation_cos": {
            "min": float(cos.min()),
            "p01": float(np.quantile(cos, 0.01)),
            "mean": float(cos.mean()),
            "frac_below_min": float((cos < cos_min).mean()),
        },
        "delta_fve_own": {"diff": d, "lo": lo, "hi": hi},
        "pass": bool(cos.min() >= cos_min and lo <= 0.0 <= hi),
        "cos_min": cos_min,
    }


def mean_norm_artifact(acts: np.ndarray, sources: np.ndarray, layer: int, tag: str) -> dict:
    norms = np.linalg.norm(acts.astype(np.float64), axis=1)
    return {
        "layer": layer,
        "dtype": tag,
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "median_norm": float(np.median(norms)),
        "by_source": {s: float(norms[sources == s].mean()) for s in SOURCES},
        "n": len(norms),
        "use": "off-layer probes rescale L12-L18 activations to mean_norm (amendment 2)",
    }


def stage_report(
    cfg: dict, prof: dict, ck: Checkpointer, items: pd.DataFrame, mse_scale: float, logger: Any
) -> dict[str, Any]:
    size = int(cfg["shard_size"])
    n = len(items)
    seed = int(cfg["seed"])
    sources = items["source"].to_numpy()
    tags = list(prof["extract_dtypes"])
    golds = {t: load_acts(ck, t, n, size) for t in tags}
    preds = {t: load_preds(ck, t, n, size) for t in tags}
    summary: dict[str, Any] = {"profile": prof, "tags": {}}
    for t in tags:
        decodes = load_decodes(ck, t, n, size)
        rep = compute_report(
            golds[t], preds[t], decodes, sources, mse_scale=mse_scale, rcfg=cfg["report"], seed=seed
        )
        _write_json(ck.local_dir / f"report-{t}.json", rep)
        summary["tags"][t] = rep
        per_row = items[["doc_id", "source", "position", "n_tokens"]].copy()
        per_row["explanation"] = [explanation_text(d) for d in decodes]
        per_row["parsed"] = [d["explanation"] is not None for d in decodes]
        per_row["cjk_fraction"] = [cjk_fraction(d["raw"]) for d in decodes]
        per_row["mse_nrm"] = mse_nrm(golds[t], preds[t], mse_scale)
        per_row["cos"] = cosine(golds[t], preds[t])
        per_row.to_parquet(ck.local_dir / f"results-{t}.parquet", index=False)
        if logger is not None:
            for name in ("overall", *SOURCES):
                r = rep[name]
                logger.log(
                    {
                        "tag": t,
                        "subset": name,
                        "n": r["n"],
                        "fve_own": r["fve_own"]["fve"],
                        "fve_own_lo": r["fve_own"]["lo"],
                        "fve_own_hi": r["fve_own"]["hi"],
                        "fve_ref": r["fve_reference"]["fve"],
                        "fve_ref_lo": r["fve_reference"]["lo"],
                        "fve_ref_hi": r["fve_reference"]["hi"],
                        "mean_cos": r["mean_cos"],
                        "parse_fail_rate": r["parse_fail_rate"],
                        "cjk_any_rate": r["cjk_any_rate"],
                        "shuffled_fve_own": rep["shuffled_control"]["fve_own"],
                    }
                )
    if len(tags) > 1:
        a, b = tags[0], tags[-1]  # e.g. fp16 vs fp32 (ground truth)
        summary["precision_check"] = {
            "a": a,
            "b": b,
            **compare_tags(
                golds[a],
                golds[b],
                preds[a],
                preds[b],
                mse_scale=mse_scale,
                cos_min=float(cfg["report"]["precision_cos_min"]),
                n_boot=int(cfg["report"]["n_boot"]),
                seed=seed,
            ),
        }
    ref_tag = tags[-1]  # highest-precision extraction available in this profile
    summary["mean_norm"] = mean_norm_artifact(golds[ref_tag], sources, int(cfg["layer"]), ref_tag)
    _write_json(ck.local_dir / f"l{cfg['layer']}_mean_norm.json", summary["mean_norm"])
    for name in ("monitor-av.json", *(f"monitor-extract-{t}.json" for t in tags)):
        p = ck.local_dir / name
        if p.exists():
            summary.setdefault("monitors", {})[name] = json.loads(p.read_text())
    _write_json(ck.local_dir / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- driver
def run(cfg: dict, profile: str, ck: Checkpointer, be: Backends, logger: Any = None) -> dict:
    prof = cfg["profiles"][profile]
    set_seed(int(cfg["seed"]))
    ck.restore()
    items_path = ck.local_dir / "items.parquet"
    if not ck.is_done("items"):
        items = build_items(cfg, be.tokenizer, be.doc_streams())
        items.to_parquet(items_path, index=False)
        ck.mark_done("items")
    items = pd.read_parquet(items_path)
    stage_extract(cfg, prof, ck, be, items)
    stage_av(cfg, prof, ck, be, len(items))
    stage_ar(cfg, prof, ck, be, len(items))
    mse_scale = float(json.loads((ck.local_dir / "mse_scale.json").read_text())["mse_scale"])
    summary = stage_report(cfg, prof, ck, items, mse_scale, logger)
    ck.mark_done("report")
    ck.finalize()
    return summary


# --------------------------------------------------------------------------- real backends
def real_backends(cfg: dict, profile: str) -> Backends:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla_audit.data.corpora import stream_parquet, ufw_docs, wildchat_docs
    from nla_audit.nla.ar import ARScorer
    from nla_audit.nla.av import AVRunner
    from nla_audit.nla.extract import load_truncated_model
    from nla_audit.nla.sidecar import NLASidecar
    from nla_audit.runtime.env import pick_dtype, provenance

    prof = cfg["profiles"][profile]
    nla_dtype = pick_dtype(prof["nla_dtype"])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])

    def streams() -> dict[str, Iterable[Doc]]:
        u, w = cfg["data"]["ufw"], cfg["data"]["wildchat"]
        return {
            "ufw": ufw_docs(stream_parquet(u["repo"], u["file"]), int(u["min_chars"])),
            "wildchat": wildchat_docs(stream_parquet(w["repo"], w["file"]), w["language"]),
        }

    def load_extractor(tag: str) -> Any:
        return load_truncated_model(cfg["base_model"], int(cfg["layer"]), pick_dtype(tag))

    def load_av() -> Any:
        d = snapshot_download(cfg["av_repo"])
        side = NLASidecar.load(d)
        if side.extraction_layer_index != int(cfg["layer"]):
            raise AssertionError(
                f"AV is for layer {side.extraction_layer_index}, not {cfg['layer']}"
            )
        model = AutoModelForCausalLM.from_pretrained(d, dtype=nla_dtype, device_map="auto")
        return AVRunner(model, AutoTokenizer.from_pretrained(d), side)

    def load_ar() -> Any:
        return ARScorer(snapshot_download(cfg["ar_repo"]), dtype=nla_dtype, device_map="auto")

    def prov(tag: str | None) -> dict[str, Any]:
        return provenance(None if tag is None else pick_dtype(tag))

    return Backends(tok, streams, load_extractor, load_av, load_ar, prov)


def _checkpointer(run_name: str, root: str, sync: bool) -> Checkpointer:
    syncer = None
    if sync:
        from nla_audit.runtime.resume import HFHubSyncer

        repo = os.environ.get("HF_ARTIFACT_REPO")
        if not repo:
            raise SystemExit("--sync needs HF_ARTIFACT_REPO (and HF_TOKEN)")
        syncer = HFHubSyncer(repo, token=os.environ.get("HF_TOKEN"))
    return Checkpointer(run_name, root, syncer=syncer)


def main(argv: list[str] | None = None) -> int:
    from nla_audit.runtime.env import load_secrets

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/step1.yaml")
    r.add_argument("--profile", required=True)
    r.add_argument("--root", default="runs")
    r.add_argument("--sync", action="store_true", help="checkpoint to private HF repo")
    c = sub.add_parser("compare", help="cross-run precision agreement, e.g. kaggle fp16 vs colab")
    c.add_argument("--config", default="configs/step1.yaml")
    c.add_argument("--a", required=True, help="run_name:tag, e.g. step1_baseline/kaggle:fp16")
    c.add_argument("--b", required=True)
    c.add_argument("--root", default="runs")
    c.add_argument("--sync", action="store_true", help="pull both runs from HF first")
    args = ap.parse_args(argv)
    load_secrets()
    cfg = load_config(args.config)

    if args.cmd == "run":
        from nla_audit.logging_utils import RunLogger

        name = f"{cfg['run_name']}/{args.profile}"
        ck = _checkpointer(name, args.root, args.sync)
        logger = RunLogger(ck.local_dir / "metrics.csv", run_name=name, config=cfg)
        try:
            summary = run(cfg, args.profile, ck, real_backends(cfg, args.profile), logger)
        finally:
            logger.close()
        print(json.dumps({t: s["verdict"] for t, s in summary["tags"].items()}, indent=1))
        if "precision_check" in summary:
            print(json.dumps(summary["precision_check"], indent=1))
        return 0

    sides = []
    for spec in (args.a, args.b):
        run_name, tag = spec.rsplit(":", 1)
        ck = _checkpointer(run_name, args.root, args.sync)
        ck.restore()
        items = pd.read_parquet(ck.local_dir / "items.parquet")
        size = int(cfg["shard_size"])
        sides.append(
            (
                items["doc_id"].to_numpy(),
                load_acts(ck, tag, len(items), size),
                load_preds(ck, tag, len(items), size),
                json.loads((ck.local_dir / "mse_scale.json").read_text())["mse_scale"],
            )
        )
    (ida, ga, pa, sa), (idb, gb, pb, _) = sides
    if not np.array_equal(ida, idb):
        raise SystemExit("runs used different items; cannot compare")
    res = compare_tags(
        ga,
        gb,
        pa,
        pb,
        mse_scale=sa,
        cos_min=float(cfg["report"]["precision_cos_min"]),
        n_boot=int(cfg["report"]["n_boot"]),
        seed=int(cfg["seed"]),
    )
    print(json.dumps({"a": args.a, "b": args.b, **res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
