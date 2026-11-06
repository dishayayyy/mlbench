import argparse
import itertools
from pathlib import Path

import pandas as pd

from mlbench.stats.bootstrap import bootstrap_ci, bootstrap_diff_ci

METRICS = ["accuracy", "auc", "f1", "log_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("device")
    args = ap.parse_args()

    # note to self: one csv per model, named <dataset>_<model>_<device>.csv
    files = sorted(Path("results").glob(f"{args.dataset}_*_{args.device}.csv"))
    if not files:
        raise SystemExit(f"no csv files found for {args.dataset} on {args.device}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # note to self: strip the dataset prefix so the table shows just logreg / mlp / xgb
    df["model"] = df["name"].str.replace(f"^{args.dataset}_", "", regex=True)
    models = list(df["model"].unique())

    print(f"\n{args.dataset} on {args.device}: mean and 95% bootstrap CI over seeds\n")
    for model in models:
        sub = df[df["model"] == model]
        parts = []
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(sub[m].to_numpy())
            parts.append(f"{m} {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        print(f"{model:8s} (n={len(sub)} seeds)  " + "   ".join(parts))

    print("\nAUC difference between models (first minus second), seeds resampled independently\n")
    for a, b in itertools.combinations(models, 2):
        va = df.loc[df["model"] == a, "auc"].to_numpy()
        vb = df.loc[df["model"] == b, "auc"].to_numpy()
        diff, lo, hi = bootstrap_diff_ci(va, vb)
        # note to self: "excludes zero" just means the whole interval is on one side of zero
        excludes = lo > 0 or hi < 0
        print(f"{a} - {b}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  excludes zero: {excludes}")

    print("\nnote: split is fixed, so these intervals reflect training randomness only, from 10 seeds")


if __name__ == "__main__":
    main()