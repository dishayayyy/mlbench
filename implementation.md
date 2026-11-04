# MLBench: Implementation Plan

Reproducible ML benchmarking along three axes: predictive performance, computational cost, statistical reliability.

Deadline: Sunday (4 working days, Thu to Sun, 5-6 focused hours per day).
Priority: polished GitHub repo first, concise technical report second.

---

## 1. Scope

| Item | Decision |
| --- | --- |
| Datasets | Adult Income (fast iteration), HIGGS (1M-row subset for training; full 11M rows only for the Pandas vs PySpark benchmark) |
| Models | Logistic Regression (in PyTorch), XGBoost, MLP (PyTorch). TabNet is optional and only attempted if everything else is done |
| Seeds | 10 per model-dataset pair |
| Bootstrap | 10,000 resamples, percentile CI (BCa optional) |
| Spark | Google Colab first; WSL2 not planned |
| Dropped | Covertype, TARNet/DragonNet (possible later extension) |

Out of scope for this week: hyperparameter search beyond a small fixed grid, multi-node Spark, publishing a paper.

---

## 2. Environment split

| Task | Where |
| --- | --- |
| Development, Adult runs, unit tests | Local (Windows 11, i5-12450HX, 12 GB RAM) |
| GPU training and GPU benchmarks | Local RTX 2050 (4 GB VRAM) and Colab GPU, always reported separately |
| PySpark vs Pandas benchmark | Colab |
| MLflow tracking | Local file store (`./mlruns`); Colab runs saved and copied back as artifacts |
| Final experiment runs | Fixed configs, run once, results committed |

Rules:
- GPU benchmarking and distributed preprocessing measure different things. Never mix them in one table.
- Record hardware info (CPU, RAM, GPU, driver, library versions) with every run so numbers from the laptop and Colab are never compared as if they were the same machine.
- 4 GB VRAM limits MLP batch size and model width on the laptop. Keep models small.

---

## 3. Architecture

```mermaid
flowchart LR
    CFG[YAML configs] --> RUN[Experiment runner]
    DATA[Dataset loaders<br/>Adult, HIGGS] --> PRE[Preprocessing<br/>Pandas / PySpark]
    PRE --> RUN
    RUN --> MODELS[Model interface<br/>LogReg, XGBoost, MLP]
    SEED[Seed utility] --> RUN
    MODELS --> EVAL[Metrics per seed]
    MODELS --> COST[Cost profiler<br/>time, memory, latency]
    EVAL --> STATS[Bootstrap CI<br/>paired comparison]
    COST --> STATS
    RUN --> MLF[MLflow<br/>parent run per experiment,<br/>child run per seed]
    STATS --> REP[Report generator<br/>tables and plots]
    MLF --> REP
```

Text version of the same flow:

```
configs -> runner -> (loader -> preprocessing) -> model -> metrics + cost
                                    |                          |
                                 MLflow  <---------------------+
                                    |
                          statistics -> plots/tables -> README + report
```

---

## 4. Repository layout

```
mlbench/
  README.md
  implementation.md
  todo.md
  pyproject.toml            # or requirements.txt
  configs/
    adult_logreg.yaml
    adult_xgb.yaml
    adult_mlp.yaml
    higgs_*.yaml
  src/mlbench/
    seed.py                 # global seeding + determinism flags
    config.py               # YAML loading and validation
    data/
      adult.py
      higgs.py
      preprocess_pandas.py
      preprocess_spark.py
    models/
      base.py               # common interface: fit, predict_proba, name
      logreg.py
      xgb.py
      mlp.py
    metrics.py              # accuracy, AUC, F1, log loss
    runner.py               # multi-seed experiment loop
    stats/
      bootstrap.py          # percentile (and optionally BCa) CIs
      compare.py            # paired model comparison
    cost/
      profiler.py           # time, throughput, memory, latency
    tracking.py             # MLflow wrappers
    report/
      plots.py
      tables.py
  scripts/
    run_experiment.py
    run_spark_benchmark.py
    make_report.py
  notebooks/
    colab_spark_benchmark.ipynb
    colab_gpu_benchmark.ipynb
  tests/
    test_seed.py
    test_bootstrap.py
    test_models_interface.py
  results/                  # final CSVs and figures (committed)
  report/
    mlbench_report.md
```

---

## 5. Component design

### 5.1 Reproducibility (seed.py)
- Seed `random`, `numpy`, `torch`, `torch.cuda`, and pass the seed to XGBoost.
- Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for GPU runs.
- Set `torch.backends.cudnn.benchmark = False`.
- Seed DataLoader workers and use a seeded `torch.Generator`.
- Document that some GPU operations stay non-deterministic or get slower in deterministic mode. Measure and report the speed cost of deterministic mode once.
- Test: the same seed run twice on CPU must give identical metrics.

### 5.2 Data
- Adult: load from OpenML or UCI, one-hot encode categoricals, standardize numerics, fixed stratified train/val/test split.
- HIGGS: UCI file (about 11M rows, 28 features, first column is the label). Convert once to Parquet. Use the first 1M rows (or a seeded sample) for training.
- Important: the data split is fixed, and only the training seed varies across the 10 runs, so variance reflects model and training randomness. State this in the report. (Optionally add a second experiment that also varies the split.)

### 5.3 Models
- One interface: `fit(X_train, y_train, X_val, y_val)`, `predict_proba(X)`.
- Logistic Regression as a single linear layer in PyTorch, so it shares the training loop with the MLP.
- MLP: 2-3 hidden layers, small width, fixed epochs with early stopping on validation.
- XGBoost: `hist` tree method, CPU and GPU (`device="cuda"`) variants.
- Fixed small hyperparameter set per model, same tuning effort for every model, written in the config.

### 5.4 Runner
- For each (dataset, model): loop over seeds 0..9, train, evaluate on test, record metrics and cost.
- Save a tidy results CSV: one row per (dataset, model, seed, device).

### 5.5 Statistics
- Percentile bootstrap CI over per-seed test metrics (10,000 resamples) for each model.
- Paired comparison between models: same seeds and same test set, bootstrap the paired difference, report the CI of the difference and whether it excludes zero.
- Also bootstrap over the test set rows for a single model to show test-set sampling uncertainty (kept separate from seed variance).
- With 10 seeds the CIs will be wide. State this explicitly.

### 5.6 Cost profiling
- Warmup iterations discarded before timing.
- `torch.cuda.synchronize()` before and after every timed GPU region.
- Metrics: training wall-clock, inference throughput (samples/s), per-batch latency (p50, p95), peak host memory, peak GPU memory (`torch.cuda.max_memory_allocated`).
- Repeat timings several times and report mean and spread, not a single run.
- Run CPU and GPU separately and record the hardware.

### 5.7 Pandas vs PySpark benchmark (Colab)
- Pipeline: read raw HIGGS, type cast, drop nulls, standardize features, write Parquet.
- Sizes: 1M, 5M, 11M rows.
- Measure: wall-clock, peak memory, and the effect of partition count (for example 4, 8, 16, 32).
- Expectation to test, not assume: 11M x 29 float64 is roughly 2.5 GB, so Pandas may fit in memory. Spark has startup and shuffle overhead, so it may be slower at this scale. Report whatever is found.
- Also compare CSV vs Parquet read time, since the file format likely matters as much as the engine.

### 5.8 MLflow
- One experiment per dataset. Parent run per (model, config), child run per seed.
- Log: params (config, seed, device), metrics (accuracy, AUC, F1, log loss), cost metrics, tags (git commit, library versions, hardware), artifacts (config file, results CSV, plots).
- Local file backend (`mlflow ui` to browse). Colab runs are exported and merged back.

---

## 6. Report outline (concise, 4-6 pages)

1. Problem and motivation
2. Methodology (datasets, models, seeds, splits)
3. Statistical approach (bootstrap, paired comparison, caveats)
4. Cost measurement protocol
5. Results: accuracy with CIs, cost vs accuracy, CPU vs GPU, Pandas vs Spark scaling
6. Limitations (few seeds, small hardware, single split, free-tier Colab variability)
7. Reproduction instructions

---

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| PySpark setup time | Colab only; pin `pyspark` version; do not touch Windows Spark |
| 4 GB VRAM limit | Small MLP, modest batch size, subset of HIGGS |
| Colab variability (shared hardware, session resets) | Record hardware per run, repeat timings, save results to Drive after every run |
| GPU non-determinism | Deterministic mode plus documented residual variance |
| Time overrun | TabNet and BCa intervals are optional; cut the second experiment (varying split) first |
| First time with MLflow and Spark | Day 1 includes a tiny end-to-end run so tool problems surface early |

---

## 8. Definition of done

- A fresh clone plus one documented command reproduces one Adult result.
- All 3 models x 2 datasets have 10 seeds logged in MLflow.
- Results table with bootstrap CIs and at least one paired comparison.
- CPU vs GPU cost table (local and Colab reported separately).
- Pandas vs Spark scaling table and plot.
- README with architecture diagram, quickstart and results summary.
- Technical report covering methodology and limitations.
