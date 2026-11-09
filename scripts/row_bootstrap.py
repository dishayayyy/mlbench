import argparse

import numpy as np
import pandas as pd

from mlbench.stats.bootstrap import bootstrap_ci
from mlbench.stats.rows import bootstrap_auc_rows
from pathlib import Path

MODELS = ["logreg", "mlp", "xgb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("device")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    probs, y_ref = {}, None
    for m in MODELS:
        d = np.load(f"results/preds/{args.dataset}_{m}_{args.device}.npz")
        # note to self: every model must have been scored on the same test labels,
        # otherwise the paired row bootstrap is meaningless
        if y_ref is None:
            y_ref = d["y_test"]
        assert np.array_equal(y_ref, d["y_test"]), f"y_test differs for {m}"
        # note to self: seed 0 predictions only, seed spread is tiny next to row spread
        probs[m] = d["preds"][0]

    per_model, diffs = bootstrap_auc_rows(y_ref, probs, n_boot=args.n_boot)

    print(f"\n{args.dataset} on {args.device}: AUC with 95% intervals, two sources of uncertainty\n")
    print(f"{'model':8s} {'point':>7s}   {'seed CI (training randomness)':32s} {'row CI (test-set sampling)':28s} width ratio")
    for m in MODELS:
        vals = pd.read_csv(f"results/{args.dataset}_{m}_{args.device}.csv")["auc"].to_numpy()
        _, s_lo, s_hi = bootstrap_ci(vals)
        point, r_lo, r_hi = per_model[m]
        ratio = (r_hi - r_lo) / (s_hi - s_lo)
        print(f"{m:8s} {point:7.4f}   [{s_lo:.4f}, {s_hi:.4f}]".ljust(52)
              + f"[{r_lo:.4f}, {r_hi:.4f}]".ljust(29) + f"{ratio:5.0f}x")

    print("\nPaired AUC differences over test rows (first minus second)\n")
    for (a, b), (point, lo, hi) in diffs.items():
        excludes = lo > 0 or hi < 0
        print(f"{a} - {b}: {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  excludes zero: {excludes}")

        # note to self: saved so make_report.py can build tables without rerunning this slow bootstrap
    pd.DataFrame([{"model": m, "point_seed0": p, "row_lo": lo, "row_hi": hi, "n_boot": args.n_boot}
                  for m, (p, lo, hi) in per_model.items()]
                 ).to_csv(Path("results") / f"rowci_{args.dataset}_{args.device}.csv", index=False)
    pd.DataFrame([{"a": a, "b": b, "diff_seed0": p, "lo": lo, "hi": hi, "n_boot": args.n_boot}
                  for (a, b), (p, lo, hi) in diffs.items()]
                 ).to_csv(Path("results") / f"rowdiff_{args.dataset}_{args.device}.csv", index=False)

    print(f"\nnote: row CI uses seed 0 predictions and {args.n_boot} resamples of {len(y_ref)} test rows")


if __name__ == "__main__":
    main()