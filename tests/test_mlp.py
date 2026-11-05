import numpy as np
from sklearn.metrics import roc_auc_score

from mlbench.models.logreg import TorchLogReg
from mlbench.models.mlp import TorchMLP
from mlbench.seed import set_seed


def _xor_data():
    # note to self: label depends on the sign of x0*x1, a linear model cannot learn this
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    return (X[:2400], y[:2400]), (X[2400:3200], y[2400:3200]), (X[3200:], y[3200:])


def test_mlp_learns_nonlinear_signal_that_logreg_cannot():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _xor_data()

    set_seed(0)
    mlp = TorchMLP(hidden_dims=(32, 32), dropout=0.0, seed=0,
                   epochs=100, lr=0.01, batch_size=128, patience=15)
    mlp.fit(Xtr, ytr, Xva, yva)
    mlp_auc = roc_auc_score(yte, mlp.predict_proba(Xte))

    set_seed(0)
    lr = TorchLogReg(seed=0, epochs=50, lr=0.01, batch_size=128)
    lr.fit(Xtr, ytr, Xva, yva)
    lr_auc = roc_auc_score(yte, lr.predict_proba(Xte))

    assert mlp_auc > 0.9
    assert lr_auc < 0.6


def test_mlp_same_seed_same_predictions():
    (Xtr, ytr), (Xva, yva), (Xte, _) = _xor_data()

    def run():
        set_seed(3)
        m = TorchMLP(hidden_dims=(16,), dropout=0.1, seed=3, epochs=5, batch_size=128)
        m.fit(Xtr, ytr, Xva, yva)
        return m.predict_proba(Xte)

    np.testing.assert_array_equal(run(), run())