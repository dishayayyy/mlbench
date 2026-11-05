import numpy as np
import pytest

from mlbench.metrics import compute_metrics


def test_known_values():
    # note to self: worked these out by hand: auc 0.75, preds [0,0,0,1] so acc 0.75, f1 = 2/3
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y, p)
    assert m["auc"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(2 / 3)


def test_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["auc"] == 1.0