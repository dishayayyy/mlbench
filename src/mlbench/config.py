from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_DATASETS = {"adult", "higgs"}
VALID_MODELS = {"logreg", "xgboost", "mlp"}
VALID_DEVICES = {"cpu", "cuda"}


@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    model: str
    seeds: list
    device: str = "cpu"
    deterministic: bool = True
    data: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)


def load_config(path) -> ExperimentConfig:
    # note to self: fail early with a clear message instead of crashing mid-experiment
    with open(Path(path), "r") as f:
        raw = yaml.safe_load(f)

    required = ["name", "dataset", "model", "seeds"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")

    cfg = ExperimentConfig(**raw)

    if cfg.dataset not in VALID_DATASETS:
        raise ValueError(f"unknown dataset '{cfg.dataset}', expected one of {VALID_DATASETS}")
    if cfg.model not in VALID_MODELS:
        raise ValueError(f"unknown model '{cfg.model}', expected one of {VALID_MODELS}")
    if cfg.device not in VALID_DEVICES:
        raise ValueError(f"unknown device '{cfg.device}', expected one of {VALID_DEVICES}")
    if not cfg.seeds:
        raise ValueError("seeds list is empty")

    return cfg