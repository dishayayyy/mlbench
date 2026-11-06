import itertools

import numpy as np
from scipy.stats import rankdata


def fast_auc(y, p):
    # note to self: rank-based AUC (Mann-Whitney), about 3x faster than sklearn and
    # fine for ties because rankdata averages tied ranks
    y = np.asarray(y)
    ranks = rankdata(p)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_auc_rows(y, probs, n_boot=10_000, alpha=0.05, seed=0):
    """Bootstrap the test rows and return AUC CIs per model plus paired differences.

    probs is a dict of model name -> 1-D test probabilities (one seed each).
    note to self: every model sees the SAME resampled rows each round, which is
    what makes the model differences paired.
    """
    y = np.asarray(y)
    names = list(probs)
    rng = np.random.default_rng(seed)
    n = len(y)

    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        for k in names:
            boot[k][b] = fast_auc(yb, np.asarray(probs[k])[idx])

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    per_model = {}
    for k in names:
        lo, hi = np.percentile(boot[k], q)
        per_model[k] = (float(fast_auc(y, probs[k])), float(lo), float(hi))

    diffs = {}
    for a, b in itertools.combinations(names, 2):
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, q)
        point = fast_auc(y, probs[a]) - fast_auc(y, probs[b])
        diffs[(a, b)] = (float(point), float(lo), float(hi))
    return per_model, diffs