import threading
import time

import numpy as np
import psutil
import torch

from mlbench.seed import set_seed


class PeakRSS:
    # note to self: polls this process's memory in a background thread and keeps the maximum.
    # one use per instance, make a new one for every timed region
    def __init__(self, interval=0.02):
        self.interval = interval
        self.proc = psutil.Process()
        self.start = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.start = self.proc.memory_info().rss
        self.peak = self.start
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def _sync(device):
    # note to self: cuda calls are asynchronous, without this I would time the launch and not the work
    if device == "cuda":
        torch.cuda.synchronize()


def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _time_training(make_model, data, device, seed, deterministic, repeats):
    # note to self: untimed warmup fit on a small slice, so cuda init and lazy imports are not in the numbers
    warm = make_model()
    warm.fit(data.X_train[:2000], data.y_train[:2000], data.X_val[:1000], data.y_val[:1000])
    _sync(device)

    secs, peaks, befores, gpu_peaks = [], [], [], []
    model = None
    for _ in range(repeats):
        set_seed(seed, deterministic=deterministic)
        model = make_model()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _sync(device)
        with PeakRSS() as mem:
            t0 = time.perf_counter()
            model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
            _sync(device)
            secs.append(time.perf_counter() - t0)
        peaks.append(mem.peak / 1e6)
        befores.append(mem.start / 1e6)
        gpu_peaks.append(torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0)
    return model, secs, peaks, befores, gpu_peaks


def _time_throughput(model, X, device, repeats, warmup=2):
    for _ in range(warmup):
        model.predict_proba(X)
    rates = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        model.predict_proba(X)
        _sync(device)
        rates.append(len(X) / (time.perf_counter() - t0))
    return rates


def _time_latency(model, X, device, batch, iters, warmup=20, seed=0):
    # note to self: random start rows with a fixed rng, so every model sees the same batches
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(X) - batch + 1, size=iters + warmup)
    ms = []
    for k, s in enumerate(starts):
        xb = X[s:s + batch]
        _sync(device)
        t0 = time.perf_counter()
        model.predict_proba(xb)
        _sync(device)
        if k >= warmup:
            ms.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def profile_model(make_model, data, device="cpu", seed=0, deterministic=True,
                  train_repeats=3, infer_repeats=5, latency_iters=200, batch_sizes=(1, 1024)):
    """Time training and inference for one model on one device. make_model builds a fresh model."""
    model, secs, peaks, befores, gpu_peaks = _time_training(
        make_model, data, device, seed, deterministic, train_repeats)

    rates = _time_throughput(model, data.X_test, device, infer_repeats)
    out = {
        "n_train": len(data.X_train), "n_test": len(data.X_test), "train_repeats": train_repeats,
        "train_s_mean": float(np.mean(secs)), "train_s_std": _std(secs),
        "peak_rss_mb": float(np.max(peaks)), "rss_before_train_mb": float(np.mean(befores)),
        "peak_gpu_mb": float(np.max(gpu_peaks)),
        "infer_samples_per_s_mean": float(np.mean(rates)), "infer_samples_per_s_std": _std(rates),
    }
    for b in batch_sizes:
        p50, p95 = _time_latency(model, data.X_test, device, b, latency_iters)
        out[f"lat_b{b}_p50_ms"] = p50
        out[f"lat_b{b}_p95_ms"] = p95
    return out