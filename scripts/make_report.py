from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # note to self: no window needed, I only save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlbench.stats.bootstrap import bootstrap_ci
from matplotlib.ticker import FuncFormatter, NullFormatter

RES = Path("results")
TAB = RES / "tables"
FIG = RES / "figures"
MODELS = ["logreg", "mlp", "xgb"]
DATASETS = ["adult", "higgs"]
METRICS = ["accuracy", "auc", "f1", "log_loss"]


def accuracy_table():
    rows = []
    for ds in DATASETS:
        rowci = pd.read_csv(RES / f"rowci_{ds}_cpu.csv").set_index("model")
        for m in MODELS:
            df = pd.read_csv(RES / f"{ds}_{m}_cpu.csv")
            r = {"dataset": ds, "model": m, "n_seeds": len(df)}
            for k in METRICS:
                mean, lo, hi = bootstrap_ci(df[k].to_numpy())
                r.update({f"{k}_mean": mean, f"{k}_std": df[k].std(ddof=1),
                          f"{k}_seed_lo": lo, f"{k}_seed_hi": hi})
            # note to self: half-width of the test-row interval, centred on the seed mean in the final table
            r["auc_row_halfwidth"] = (rowci.loc[m, "row_hi"] - rowci.loc[m, "row_lo"]) / 2
            rows.append(r)
    return pd.DataFrame(rows)


def cost_table():
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            c = pd.read_csv(RES / f"cost_{ds}_{m}_cpu.csv").iloc[0]
            rows.append({"dataset": ds, "model": m, **c[[
                "train_s_mean", "train_s_std", "peak_rss_mb", "infer_samples_per_s_mean",
                "lat_b1_p50_ms", "lat_b1_p95_ms", "lat_b1024_p50_ms", "lat_b1024_p95_ms"]].to_dict()})
    return pd.DataFrame(rows)


def diff_table():
    parts = []
    for ds in DATASETS:
        d = pd.read_csv(RES / f"rowdiff_{ds}_cpu.csv")
        d.insert(0, "dataset", ds)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def spark_tables():
    sp = pd.read_csv(RES / "spark_vs_pandas.csv")
    # note to self: if a cell was run twice in colab, the last run wins
    sp = sp.drop_duplicates(subset=["engine", "size", "partitions"], keep="last")
    return sp[sp["partitions"] == 0], sp[sp["partitions"] > 0]


def plot_accuracy(acc):
    # note to self: third panel zooms in on mlp vs xgb on higgs, otherwise the logreg gap squashes them flat
    panels = [("adult", MODELS, "adult"), ("higgs", MODELS, "higgs"),
              ("higgs", ["mlp", "xgb"], "higgs (zoom: mlp vs xgb)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (ds, models, title) in zip(axes, panels):
        sub = acc[acc["dataset"] == ds].set_index("model").loc[models]
        x = np.arange(len(models))
        ax.errorbar(x, sub["auc_mean"], yerr=sub["auc_row_halfwidth"], fmt="o", capsize=4,
                    label="seed mean, bar = test-row 95% CI")
        for i, m in enumerate(models):
            vals = pd.read_csv(RES / f"{ds}_{m}_cpu.csv")["auc"]
            ax.scatter(np.full(len(vals), i) + 0.1, vals, s=8, alpha=0.6, color="gray",
                       label="single seeds" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("test AUC")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    return fig


def plot_cost_accuracy(acc, cost):
    ticks = {"adult": [1, 2, 4, 8], "higgs": [20, 40, 80, 160, 320]}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, ds in zip(axes, DATASETS):
        for m in MODELS:
            a = acc[(acc["dataset"] == ds) & (acc["model"] == m)].iloc[0]
            c = cost[(cost["dataset"] == ds) & (cost["model"] == m)].iloc[0]
            ax.errorbar(c["train_s_mean"], a["auc_mean"], xerr=c["train_s_std"], fmt="o", capsize=3)
            ax.annotate(m, (c["train_s_mean"], a["auc_mean"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xscale("log")
        # note to self: explicit ticks and no minor labels, the default log labels overlapped each other
        ax.set_xticks(ticks[ds])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training time (s, CPU, log scale)")
        ax.set_ylabel("test AUC")
        ax.set_title(ds)
    fig.tight_layout()
    return fig


def plot_spark(main):
    order = {"1m": 1, "5m": 5, "11m": 11}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for eng in ["pandas", "spark"]:
        d = main[main["engine"] == eng].copy()
        d["rows_m"] = d["size"].map(order)
        d = d.sort_values("rows_m")
        axes[0].plot(d["rows_m"], d["wall_s"], marker="o", label=eng)
        axes[1].plot(d["rows_m"], d["peak_gb"], marker="o", label=eng)
    axes[0].set_ylabel("wall time (s)")
    axes[1].set_ylabel("peak memory (GB)")
    for ax in axes:
        ax.set_xlabel("rows (millions)")
        ax.legend()
    fig.tight_layout()
    return fig


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    acc, cost, diffs = accuracy_table(), cost_table(), diff_table()
    sp_main, sp_sweep = spark_tables()
    fmt = pd.read_csv(RES / "csv_vs_parquet.csv")

    for name, t in [("accuracy", acc), ("cost", cost), ("model_diffs", diffs),
                    ("spark_vs_pandas", sp_main), ("spark_partitions", sp_sweep), ("csv_vs_parquet", fmt)]:
        t.to_csv(TAB / f"{name}.csv", index=False)

    for name, fig in [("accuracy", plot_accuracy(acc)), ("cost_vs_accuracy", plot_cost_accuracy(acc, cost)),
                      ("spark_scaling", plot_spark(sp_main))]:
        fig.savefig(FIG / f"{name}.png", dpi=200)
        fig.savefig(FIG / f"{name}.pdf")  # note to self: pdf is what goes into the latex report
        plt.close(fig)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nACCURACY (seed mean, seed std, test-row CI half-width for AUC)")
    print(acc[["dataset", "model", "n_seeds", "auc_mean", "auc_std", "auc_row_halfwidth",
               "accuracy_mean", "accuracy_std", "f1_mean", "log_loss_mean"]].round(4).to_string(index=False))
    print("\nCOST (cpu)")
    print(cost.round(3).to_string(index=False))
    print("\nMODEL DIFFERENCES over test rows (seed 0 predictions)")
    print(diffs.round(4).to_string(index=False))
    print("\nPANDAS VS SPARK")
    print(sp_main[["engine", "size", "status", "wall_s", "peak_gb", "startup_s", "pipeline_s"]].to_string(index=False))
    print("\nPARTITION SWEEP (1M rows)")
    print(sp_sweep[["partitions", "status", "wall_s", "peak_gb"]].to_string(index=False))
    print("\nCSV VS PARQUET")
    print(fmt.to_string(index=False))
    print(f"\nwrote tables to {TAB} and figures to {FIG}")


if __name__ == "__main__":
    main()