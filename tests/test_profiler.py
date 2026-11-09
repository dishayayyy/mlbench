import numpy as np

from mlbench.cost.profiler import profile_model
from mlbench.data.base import TabularSplit
from mlbench.models.base import BaseModel


class _Dummy(BaseModel):
    # note to self: predicts the training base rate, just enough to exercise the profiler
    name = "dummy"

    def fit(self, X_train, y_train, X_val, y_val):
        self.p = float(np.mean(y_train))
        return {}

    def predict_proba(self, X):
        return np.full(len(X), self.p)


def _data(n=3000, d=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return TabularSplit(X, y, X[:500], y[:500], X[:2000], y[:2000], feature_names=[f"f{i}" for i in range(d)])


def test_profile_model_returns_sane_numbers():
    out = profile_model(_Dummy, _data(), train_repeats=2, infer_repeats=3, latency_iters=30)
    assert out["train_s_mean"] > 0
    assert out["infer_samples_per_s_mean"] > 0
    assert out["peak_rss_mb"] >= out["rss_before_train_mb"] > 0
    # note to self: a percentile pair can never be reversed, if this fails the latency code is broken
    assert out["lat_b1_p95_ms"] >= out["lat_b1_p50_ms"] > 0
    assert out["lat_b1024_p95_ms"] >= out["lat_b1024_p50_ms"] > 0
    assert out["peak_gpu_mb"] == 0