import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlbench.stats.rows import bootstrap_auc_rows, fast_auc


def _toy(seed=0, n=500):
    # note to self: noisy scores that are informative but not perfect
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p = np.clip(0.3 * y + rng.normal(0.4, 0.25, size=n), 0, 1)
    return y, p


def test_fast_auc_matches_sklearn():
    # note to self: if this ever fails my rank formula is wrong, nothing else matters
    y, p = _toy()
    assert fast_auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_row_ci_brackets_point_estimate():
    y, p = _toy()
    per_model, _ = bootstrap_auc_rows(y, {"m": p}, n_boot=300)
    point, lo, hi = per_model["m"]
    assert lo <= point <= hi


def test_identical_models_have_zero_difference():
    # note to self: same predictions on the same rows must give exactly zero, no noise
    y, p = _toy()
    _, diffs = bootstrap_auc_rows(y, {"a": p, "b": p.copy()}, n_boot=100)
    point, lo, hi = diffs[("a", "b")]
    assert point == 0 and lo == 0 and hi == 0