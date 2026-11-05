import platform
import subprocess

import mlflow
import numpy as np
import psutil
import sklearn
import torch
import xgboost

# note to self: one local sqlite file at the project root, gitignored through *.db
DEFAULT_URI = "sqlite:///mlflow.db"


def init_tracking(experiment: str, uri: str = DEFAULT_URI) -> None:
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def _git_commit() -> str:
    # note to self: "-dirty" means I ran with uncommitted changes, so the commit alone can't reproduce the run
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_tags(device: str) -> dict:
    # note to self: results from the laptop and colab must never be compared blindly, so tag the hardware
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_gb": str(round(psutil.virtual_memory().total / 1e9, 1)),
        "gpu": gpu,
        "device_requested": device,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }


def flatten_params(cfg) -> dict:
    # note to self: mlflow params are flat key/value pairs, so nested dicts get a prefix
    params = {
        "name": cfg.name,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "device": cfg.device,
        "deterministic": cfg.deterministic,
        "n_seeds": len(cfg.seeds),
    }
    for prefix, section in (("data", cfg.data), ("model", cfg.model_params), ("train", cfg.train)):
        for k, v in section.items():
            params[f"{prefix}.{k}"] = v
    return params


def log_config(cfg) -> None:
    mlflow.log_params(flatten_params(cfg))