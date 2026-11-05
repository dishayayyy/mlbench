# MLBench: Todo

Plan: 4 days, about 5-6 focused hours per day. Check boxes as you go.
Cut order if time runs short: TabNet, BCa intervals, varying-split experiment, extra Spark partition sweeps.
Last updated: Thursday 2026-11-05, afternoon. Code marked "sent, not yet applied" was given in chat but not confirmed as run or committed.

---

## Thursday: Foundation (done)

Setup
- [x] Create GitHub repo `mlbench` (public), add `.gitignore`
- [x] Create virtual environment, install torch (CUDA, GPU detected), numpy, pandas, scikit-learn, xgboost, mlflow, pyyaml, pytest
- [x] Create the folder layout
- [x] Record library versions in `requirements-lock.txt` (to be cleaned on Sunday)

Core utilities
- [x] `seed.py`: seed all libraries, deterministic flags
- [x] Test: same seed twice gives identical result on CPU
- [x] `config.py` and example YAMLs (logreg, mlp, xgb)

Data and models
- [x] Adult loader with fixed stratified train/val/test split
- [x] Base model interface and shared PyTorch trainer
- [x] Logistic Regression (PyTorch)
- [x] MLP (PyTorch)
- [x] XGBoost wrapper
- [x] Model factory
- [x] `metrics.py` (accuracy, AUC, F1, log loss)

First tracked run
- [x] MLflow tracking wrapper
- [x] Run all three models on Adult with 10 seeds, logged to MLflow (development runs)
- [ ] Open the MLflow UI and confirm the parent run shows the git commit and GPU tags (Model training view, Experiments, adult)
- [x] Commit and push (last confirmed commit: 1a52547)

---

## Friday: Statistics and cost (target: Adult fully benchmarked, cost profiler working)

Finish the runner and statistics (start here)
- [ ] Replace `scripts/run_experiment.py` with the version that also writes `results/<name>_<device>.csv` and `results/preds/<name>_<device>.npz` (sent, not yet applied)
- [ ] `stats/bootstrap.py` with `bootstrap_ci` and `bootstrap_diff_ci` plus `tests/test_bootstrap.py`, expect 23 tests passing (sent, not yet applied)
- [ ] `scripts/summarize.py` printing CIs per model and AUC differences (sent, not yet applied)
- [ ] Rerun the three Adult configs; seed 0 must reproduce the earlier numbers (logreg AUC 0.9046, MLP 0.9114, XGBoost 0.9287)
- [ ] Bootstrap over test rows using the saved predictions; report separately from seed variance
- [ ] Move the multi-seed loop into `src/mlbench/runner.py`
- [ ] Update this file and commit

Cost profiler
- [ ] Warmup plus timing with `torch.cuda.synchronize()`
- [ ] Training time, inference throughput, latency p50/p95
- [ ] Peak host memory and peak GPU memory
- [ ] XGBoost GPU: keep data on the GPU while timing prediction
- [ ] Run on CPU and local GPU for Adult

HIGGS
- [ ] Download HIGGS, convert to Parquet
- [ ] HIGGS loader: 1M-row seeded random sample (fixed sample seed), same fixed split procedure as Adult
- [ ] One smoke run per model on HIGGS

---

## Saturday: Spark and hardware benchmarks (target: all final experiments launched)

PySpark on Colab
- [ ] Colab notebook: install pyspark, confirm Java works, load HIGGS
- [ ] Spark preprocessing pipeline (cast, clean, standardize, write Parquet)
- [ ] Pandas equivalent pipeline
- [ ] Benchmark at 1M, 5M, 11M rows
- [ ] Partition count sweep (4, 8, 16, 32)
- [ ] CSV vs Parquet read comparison
- [ ] Save results to Drive and copy back to repo

GPU benchmarks
- [ ] Local GPU benchmark (RTX 2050)
- [ ] Colab GPU benchmark (separate table)
- [ ] Deterministic vs non-deterministic mode speed cost, measured once

Final runs
- [ ] Delete `mlflow.db` so development runs do not mix with final results
- [ ] Commit everything first so the git commit tag has no `-dirty` suffix
- [ ] Final 10-seed runs for all 3 models on Adult and the HIGGS subset
- [ ] Verify every run appears in MLflow with hardware and version tags
- [ ] Commit results CSVs

---

## Sunday: Packaging (target: repo and report ready)

Plots and tables
- [ ] Accuracy and AUC with bootstrap CIs (seed variance and test-row variance shown separately)
- [ ] Cost vs accuracy trade-off plot
- [ ] CPU vs GPU table
- [ ] Pandas vs Spark scaling plot
- [ ] Save to `results/`

Documentation
- [ ] README: overview, architecture diagram (Mermaid from implementation.md), quickstart, results summary
- [ ] Technical report (4-6 pages): methodology, results, limitations
- [ ] Add limitations honestly (few seeds, single split, free Colab, small VRAM)
- [ ] Delete `scripts/quick_compare.py` or move it into a clearly labelled scratch folder

Verification
- [ ] Clean `requirements-lock.txt` (remove the editable-install line, add the CUDA index URL note for torch)
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
