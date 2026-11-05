from mlbench.config import ExperimentConfig
from mlbench.models.logreg import TorchLogReg
from mlbench.models.mlp import TorchMLP
from mlbench.models.xgb import XGBModel


def build_model(cfg: ExperimentConfig, seed: int):
    # note to self: the runner only ever calls this, it never imports a model class directly
    if cfg.model == "logreg":
        return TorchLogReg(seed=seed, device=cfg.device, **cfg.train)
    if cfg.model == "mlp":
        return TorchMLP(seed=seed, device=cfg.device, **cfg.model_params, **cfg.train)
    if cfg.model == "xgboost":
        return XGBModel(seed=seed, device=cfg.device, **cfg.model_params)
    raise ValueError(f"no model registered for '{cfg.model}'")