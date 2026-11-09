import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from mlbench import tracking
from mlbench.config import load_config
from mlbench.cost.profiler import profile_model
from mlbench.models.factory import build_model
from run_experiment import load_data  # note to self: works because python adds scripts/ to the import path

RESULTS_DIR = Path("results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override the device in the config, e.g. cuda")
    ap.add_argument("--train-repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    data = load_data(cfg)

    # note to self: seed 0 for every repeat on purpose, this measures timing noise, not accuracy variance
    stats = profile_model(lambda: build_model(cfg, 0), data, device=cfg.device, seed=0,
                          deterministic=cfg.deterministic, train_repeats=args.train_repeats)
    env = {f"env_{k}": v for k, v in tracking.env_tags(cfg.device).items()}
    row = {"dataset": cfg.dataset, "name": cfg.name, "device": cfg.device, **stats, **env}

    out = RESULTS_DIR / f"cost_{cfg.name}_{cfg.device}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    show = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}
    print(f"{cfg.name} on {cfg.device}:")
    for k, v in show.items():
        print(f"  {k}: {v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()