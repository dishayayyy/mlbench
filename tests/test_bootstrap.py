import numpy as np
import pytest

from mlbench.stats.bootstrap import bootstrap_ci, bootstrap_diff_ci


def test_ci_contains_mean():
    # note to self: the percentile interval should always bracket the sample mean here
    vals = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
    mean, lo, hi = bootstrap_ci(vals)
    assert lo <= mean <= hi
    assert mean == pytest.approx(0.92)


def test_ci_constant_values_has_zero_width():
    # note to self: if every seed gives the same number there is no uncertainty to report
    mean, lo, hi = bootstrap_ci([0.5] * 10)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_same_seed_gives_same_ci():
    # note to self: the bootstrap itself has to be reproducible, not just the models
    vals = [0.90, 0.92, 0.91, 0.95, 0.93]
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)


def test_diff_ci_excludes_zero_for_separated_groups():
    # note to self: groups that clearly differ should give a CI that does not touch zero
    diff, lo, hi = bootstrap_diff_ci([0.93, 0.931, 0.929], [0.90, 0.901, 0.899])
    assert diff > 0
    assert lo > 0