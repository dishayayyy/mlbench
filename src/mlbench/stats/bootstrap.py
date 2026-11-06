import numpy as np

# note to self: both functions use the percentile method, so the CI is just
# the 2.5th and 97.5th percentile of the resampled statistic (for alpha=0.05)


def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of a 1-D array of values."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)  # note to self: own rng so this never touches global state

    # note to self: draw all resamples at once, shape (n_boot, n_values), then average each row
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot=10_000, alpha=0.05, seed=0):
    """CI for mean(a) - mean(b), resampling each group independently.

    note to self: seed k of model A has no real link to seed k of model B,
    so these are NOT paired. Each group gets its own resample indices.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)

    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)