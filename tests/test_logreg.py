import numpy as np
from sklearn.metrics import roc_auc_score

from mlbench.models.logreg import TorchLogReg
from mlbench.seed import set_seed


def _data():
    # note to self: synthetic data with a known linear signal, so a working model must score high
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    w = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
    y = ((X @ w + rng.normal(scale=0.5, size=2000)) > 0).astype(np.int64)
    return (X[:1200], y[:1200]), (X[1200:1500], y[1200:1500]), (X[1500:], y[1500:])


def _fit_predict(seed):
    set_seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, _) = _data()
    m = TorchLogReg(seed=seed, epochs=30, lr=0.05, batch_size=64)
    m.fit(Xtr, ytr, Xva, yva)
    return m.predict_proba(Xte)


def test_learns_the_signal():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _data()
    set_seed(0)
    m = TorchLogReg(seed=0, epochs=30, lr=0.05, batch_size=64)
    m.fit(Xtr, ytr, Xva, yva)
    assert roc_auc_score(yte, m.predict_proba(Xte)) > 0.9


def test_same_seed_same_predictions():
    np.testing.assert_array_equal(_fit_predict(7), _fit_predict(7))


def test_different_seed_different_predictions():
    assert not np.array_equal(_fit_predict(1), _fit_predict(2))