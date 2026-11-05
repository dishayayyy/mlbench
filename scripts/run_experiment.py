import argparse

import mlflow
import numpy as np

from mlbench import tracking
from mlbench.config import load_config
from mlbench.data.adult import load_adult
from mlbench.metrics import compute_metrics
from mlbench.models.factory import build_model
from mlbench.seed import set_seed


def load_data(cfg):
    # note to self: higgs gets added here later
    if cfg.dataset == "adult":
        return load_adult(**cfg.data)
    raise NotImplementedError(f"dataset '{cfg.dataset}' not wired up yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=None, help="use only the first N seeds, for quick tests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.seeds[: args.n_seeds] if args.n_seeds else cfg.seeds
    data = load_data(cfg)

    tracking.init_tracking(cfg.dataset)
    with mlflow.start_run(run_name=cfg.name):
        tracking.log_config(cfg)
        mlflow.set_tags(tracking.env_tags(cfg.device))
        mlflow.log_artifact(args.config)

        per_seed = []
        for seed in seeds:
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
                set_seed(seed, deterministic=cfg.deterministic)
                model = build_model(cfg, seed)
                info = model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
                metrics = compute_metrics(data.y_test, model.predict_proba(data.X_test))

                mlflow.log_param("seed", seed)
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({k: float(v) for k, v in info.items()})
                per_seed.append(metrics)
                print(f"seed {seed}: auc {metrics['auc']:.4f}  acc {metrics['accuracy']:.4f}")

        # note to self: plain mean/std across seeds for now, bootstrap cis come tomorrow
        for key in per_seed[0]:
            vals = np.array([m[key] for m in per_seed])
            mlflow.log_metric(f"{key}_mean", float(vals.mean()))
            mlflow.log_metric(f"{key}_std", float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


if __name__ == "__main__":
    main()