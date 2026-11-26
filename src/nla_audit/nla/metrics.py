"""Reconstruction metrics, exactly as defined in the official docs/inference.md.

Both gold and predicted vectors are L2-normalised to `mse_scale` (= sqrt(d)), so
per-vector MSE = 2(1 - cos) in [0, 4]. FVE uses the UN-normalised mean of the
normalised golds in the denominator (normalising the mean inflates FVE).

The released Qwen L20 number (fve_nrm 0.752) used a FIXED training-set denominator
(0.7335), so we always report FVE against both that constant and our own sample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Qwen L20 training-set Var(v_nrm), from examples/qwen7b_layer20_step4200.txt:31 of the
# official repo. This is a published reference number, not an NLA runtime constant
# (those come from nla_meta.yaml), so it lives in the config of each run that uses it.


def normalize_rows(x: np.ndarray, scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None) * scale


def mse_nrm(gold: np.ndarray, pred: np.ndarray, mse_scale: float) -> np.ndarray:
    """Per-row direction MSE (= 2(1-cos))."""
    g = normalize_rows(gold, mse_scale)
    p = normalize_rows(pred, mse_scale)
    return ((p - g) ** 2).mean(axis=-1)


def cosine(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    g = normalize_rows(gold, 1.0)
    p = normalize_rows(pred, 1.0)
    return (g * p).sum(axis=-1)


def baseline_variance(gold: np.ndarray, mse_scale: float) -> float:
    """Var of normalised golds around their UN-normalised mean (the canonical denominator)."""
    g = normalize_rows(gold, mse_scale)
    mu = g.mean(axis=0, keepdims=True)  # do NOT re-normalise mu
    return float(((g - mu) ** 2).mean())


def fve(
    gold: np.ndarray, pred: np.ndarray, mse_scale: float, denominator: float | None = None
) -> float:
    """1 - mean(mse_nrm) / denominator. denominator=None → computed from `gold`."""
    num = float(mse_nrm(gold, pred, mse_scale).mean())
    den = baseline_variance(gold, mse_scale) if denominator is None else float(denominator)
    return 1.0 - num / den


class FVEResampler:
    """`fve(gold[idx], pred[idx], ...)` for many index resamples, without re-normalising.

    Normalises once, then each call needs only two weighted sums over the rows:
    num = Σ w·mse / n and Var = Σ w·|g|²/(n·d) − |μ|²/d with μ = Σ w·g / n.
    """

    def __init__(self, gold: np.ndarray, pred: np.ndarray, mse_scale: float) -> None:
        self.g = normalize_rows(gold, mse_scale)
        self.mse = mse_nrm(gold, pred, mse_scale)
        self.sq = (self.g**2).sum(axis=-1)
        self.n, self.d = self.g.shape

    def __call__(self, idx: np.ndarray, denominator: float | None = None) -> float:
        w = np.bincount(idx, minlength=self.n).astype(np.float64)
        m = w.sum()
        num = float(w @ self.mse) / m
        if denominator is None:
            mu = (w @ self.g) / m
            den = float(w @ self.sq) / (m * self.d) - float(mu @ mu) / self.d
        else:
            den = float(denominator)
        return 1.0 - num / den


def bootstrap_ci(
    n: int,
    stat: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over row indices. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)
    point = stat(idx_all)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat(rng.integers(0, n, size=n))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def hierarchical_bootstrap_ci(
    groups: Sequence[np.ndarray],
    stat: Callable[[list[np.ndarray]], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-level bootstrap: resample groups (e.g. training seeds), then items within each.

    `groups[i]` holds per-item values for group i; `stat` maps a list of resampled
    groups to a scalar.
    """
    rng = np.random.default_rng(seed)
    point = stat([np.asarray(g) for g in groups])
    boots = np.empty(n_boot)
    k = len(groups)
    for b in range(n_boot):
        picked = rng.integers(0, k, size=k)
        res = []
        for gi in picked:
            g = np.asarray(groups[gi])
            res.append(g[rng.integers(0, len(g), size=len(g))])
        boots[b] = stat(res)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
