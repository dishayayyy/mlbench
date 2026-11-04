# MLBench: Todo

Plan: 4 days, about 5-6 focused hours per day. Check boxes as you go.
Cut order if time runs short: TabNet, BCa intervals, varying-split experiment, extra Spark partition sweeps.

---

## Thursday: Foundation (target: one working end-to-end run)

Setup (about 1 h)
- [ ] Create GitHub repo `mlbench`, add `.gitignore` (mlruns, data, checkpoints)
- [ ] Create virtual environment, install torch, numpy, pandas, scikit-learn, xgboost, mlflow, pyyaml, pytest
- [ ] Create the folder layout from implementation.md
- [ ] Record library versions in requirements file

Core utilities (about 1.5 h)
- [ ] `seed.py`: seed all libraries, deterministic flags
- [ ] Test: same seed twice gives identical result on CPU
- [ ] `config.py` and one example YAML

Data and models (about 2 h)
- [ ] Adult loader with fixed stratified train/val/test split
- [ ] Base model interface
- [ ] Logistic Regression (PyTorch)
- [ ] MLP (PyTorch)
- [ ] XGBoost wrapper
- [ ] `metrics.py` (accuracy, AUC, F1, log loss)

First tracked run (about 1 h)
- [ ] Wrapper for MLflow logging
- [ ] Run one model on Adult with one seed, view it in `mlflow ui`
- [ ] Commit and push

---

## Friday: Statistics and cost (target: Adult fully benchmarked)

Runner (about 1.5 h)
- [ ] Multi-seed runner (10 seeds), parent run per experiment, child run per seed
- [ ] Save tidy results CSV
- [ ] Run all 3 models on Adult

Statistics (about 1.5 h)
- [ ] Percentile bootstrap CI (10,000 resamples)
- [ ] Unit test the bootstrap on a known distribution
- [ ] Paired model comparison (CI of the difference)
- [ ] Separate bootstrap over test rows vs across seeds

Cost profiler (about 1.5 h)
- [ ] Warmup plus timing with `cuda.synchronize()`
- [ ] Training time, inference throughput, latency p50/p95
- [ ] Peak host memory and peak GPU memory
- [ ] Run on CPU and local GPU for Adult

HIGGS (about 1 h)
- [ ] Download HIGGS, convert to Parquet
- [ ] HIGGS loader with 1M-row training subset
- [ ] One smoke run per model on HIGGS

---

## Saturday: Spark and hardware benchmarks (target: all final experiments launched)

PySpark on Colab (about 2.5 h)
- [ ] Colab notebook: install pyspark, confirm Java works, load HIGGS
- [ ] Spark preprocessing pipeline (cast, clean, standardize, write Parquet)
- [ ] Pandas equivalent pipeline
- [ ] Benchmark at 1M, 5M, 11M rows
- [ ] Partition count sweep (4, 8, 16, 32)
- [ ] CSV vs Parquet read comparison
- [ ] Save results to Drive and copy back to repo

GPU benchmarks (about 1.5 h)
- [ ] Local GPU benchmark (RTX 2050)
- [ ] Colab GPU benchmark (separate table)
- [ ] Deterministic vs non-deterministic mode speed cost, measured once

Final runs (about 1.5 h)
- [ ] Final 10-seed runs for all 3 models on Adult and HIGGS subset
- [ ] Verify every run appears in MLflow with hardware and version tags
- [ ] Commit results CSVs

---

## Sunday: Packaging (target: repo and report ready)

Plots and tables (about 1.5 h)
- [ ] Accuracy and AUC with bootstrap CIs
- [ ] Cost vs accuracy trade-off plot
- [ ] CPU vs GPU table
- [ ] Pandas vs Spark scaling plot
- [ ] Save to `results/`

Documentation (about 2 h)
- [ ] README: overview, architecture diagram, quickstart, results summary
- [ ] Technical report (4-6 pages): methodology, results, limitations
- [ ] Add limitations honestly (few seeds, single split, free Colab, small VRAM)

Verification (about 1 h)
- [ ] Fresh clone in a clean environment reproduces one Adult result
- [ ] Run tests
- [ ] Clean up code, add docstrings to public functions
- [ ] Final commit and tag `v0.1`

---

## Optional extras (only if ahead of schedule)
- [ ] TabNet as a fourth model
- [ ] BCa bootstrap intervals
- [ ] Second experiment that also varies the data split
- [ ] Simple GitHub Actions workflow running the tests
