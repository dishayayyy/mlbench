import numpy as np
from sklearn.metrics import roc_auc_score

from mlbench.models.xgb import XGBModel


def _xor_data():
    # note to self: same nonlinear data as the mlp test, trees should handle it easily
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    return (X[:2400], y[:2400]), (X[2400:3200], y[2400:3200]), (X[3200:], y[3200:])


def test_xgb_learns_nonlinear_signal():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _xor_data()
    m = XGBModel(seed=0, max_depth=3, n_estimators=200, n_jobs=1)
    m.fit(Xtr, ytr, Xva, yva)
    assert roc_auc_score(yte, m.predict_proba(Xte)) > 0.9


def _run(seed):
    (Xtr, ytr), (Xva, yva), (Xte, _) = _xor_data()
    m = XGBModel(seed=seed, max_depth=3, n_estimators=50, n_jobs=1)
    m.fit(Xtr, ytr, Xva, yva)
    return m.predict_proba(Xte)


def test_xgb_same_seed_same_predictions():
    np.testing.assert_array_equal(_run(5), _run(5))


def test_xgb_different_seed_different_predictions():
    # note to self: this is the check that subsample/colsample below 1.0 really injects randomness
    assert not np.array_equal(_run(1), _run(2))