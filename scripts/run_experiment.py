import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mlbench import tracking
from mlbench.config import load_config
from mlbench.data.adult import load_adult
from mlbench.metrics import compute_metrics
from mlbench.models.factory import build_model
from mlbench.seed import set_seed
from mlbench.data.higgs import load_higgs

RESULTS_DIR = Path("results")


def load_data(cfg):
    if cfg.dataset == "adult":
        return load_adult(**cfg.data)
    if cfg.dataset == "higgs":
        return load_higgs(**cfg.data)
    raise NotImplementedError(f"dataset '{cfg.dataset}' not wired up yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=None, help="use only the first N seeds, for quick tests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.seeds[: args.n_seeds] if args.n_seeds else cfg.seeds
    data = load_data(cfg)

    # note to self: file stem is name + device so cpu and gpu runs never overwrite each other
    stem = f"{cfg.name}_{cfg.device}"
    (RESULTS_DIR / "preds").mkdir(parents=True, exist_ok=True)

    tracking.init_tracking(cfg.dataset)
    with mlflow.start_run(run_name=cfg.name):
        tracking.log_config(cfg)
        mlflow.set_tags(tracking.env_tags(cfg.device))
        mlflow.log_artifact(args.config)

        rows = []
        all_preds = []  # note to self: one row of test probabilities per seed
        for seed in seeds:
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
                set_seed(seed, deterministic=cfg.deterministic)
                model = build_model(cfg, seed)
                info = model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
                probs = np.asarray(model.predict_proba(data.X_test), dtype=np.float64)
                metrics = compute_metrics(data.y_test, probs)

                mlflow.log_param("seed", seed)
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({k: float(v) for k, v in info.items()})

                rows.append(
                    {"dataset": cfg.dataset, "name": cfg.name, "device": cfg.device, "seed": seed,
                     **metrics, **{k: float(v) for k, v in info.items()}}
                )
                all_preds.append(probs)
                print(f"seed {seed}: auc {metrics['auc']:.4f}  acc {metrics['accuracy']:.4f}")

        df = pd.DataFrame(rows)

        # note to self: mean/std across seeds stays in mlflow, the real cis are computed from the csv
        for key in ["accuracy", "auc", "f1", "log_loss"]:
            vals = df[key].to_numpy()
            mlflow.log_metric(f"{key}_mean", float(vals.mean()))
            mlflow.log_metric(f"{key}_std", float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

        csv_path = RESULTS_DIR / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

        # note to self: y_test is saved once here because the test-row bootstrap needs it next to the preds
        np.savez_compressed(
            RESULTS_DIR / "preds" / f"{stem}.npz",
            preds=np.stack(all_preds),
            y_test=np.asarray(data.y_test),
            seeds=np.asarray(seeds),
        )
        mlflow.log_artifact(str(csv_path))
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()