from __future__ import annotations

import numpy as np
import pytest

from nla_audit.nla.metrics import (
    baseline_variance,
    bootstrap_ci,
    cosine,
    fve,
    hierarchical_bootstrap_ci,
    mse_nrm,
    normalize_rows,
)

D = 32
SCALE = D**0.5


@pytest.fixture
def gold():
    rng = np.random.default_rng(0)
    # anisotropic with a shared mean direction, like real residual streams
    return rng.normal(size=(500, D)) + 3.0 * np.ones(D)


def test_mse_is_two_one_minus_cos(gold):
    rng = np.random.default_rng(1)
    pred = gold + rng.normal(scale=2.0, size=gold.shape)
    np.testing.assert_allclose(mse_nrm(gold, pred, SCALE), 2 * (1 - cosine(gold, pred)), atol=1e-9)


def test_mse_invariant_to_pred_norm(gold):
    pred = gold * 0.3 + 1e-3
    np.testing.assert_allclose(mse_nrm(gold, pred, SCALE), mse_nrm(gold, pred * 50, SCALE))


def test_fve_perfect_and_mean(gold):
    assert fve(gold, gold, SCALE) == pytest.approx(1.0)
    g = normalize_rows(gold, SCALE)
    mean_pred = np.repeat(g.mean(axis=0, keepdims=True), len(g), axis=0)
    # predicting the (direction of the) mean gives FVE close to 0 (not exactly: the
    # metric renormalises predictions onto the sphere)
    assert abs(fve(gold, mean_pred, SCALE)) < 0.15


def test_denominator_uses_unnormalised_mean(gold):
    g = normalize_rows(gold, SCALE)
    mu = g.mean(axis=0, keepdims=True)
    wrong = float(((g - normalize_rows(mu, SCALE)) ** 2).mean())
    right = baseline_variance(gold, SCALE)
    assert right < wrong  # normalising mu inflates the denominator (official footgun #2)


def test_fixed_denominator(gold):
    rng = np.random.default_rng(2)
    pred = gold + rng.normal(size=gold.shape)
    num = mse_nrm(gold, pred, SCALE).mean()
    assert fve(gold, pred, SCALE, denominator=0.7335) == pytest.approx(1 - num / 0.7335)


def test_bootstrap_ci_brackets_point():
    x = np.random.default_rng(3).normal(size=400)
    point, lo, hi = bootstrap_ci(len(x), lambda idx: float(x[idx].mean()), n_boot=500)
    assert lo <= point <= hi
    assert hi - lo < 0.3


def test_hierarchical_bootstrap():
    rng = np.random.default_rng(4)
    groups = [rng.normal(loc=m, size=60) for m in (0.4, 0.5, 0.6)]
    point, lo, hi = hierarchical_bootstrap_ci(
        groups, lambda gs: float(np.mean([g.mean() for g in gs])), n_boot=500
    )
    assert lo <= point <= hi
