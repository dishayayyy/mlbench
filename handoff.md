# MLBench: Handoff Document

Written: Thursday 2026-11-05, about 13:00 IST. Project deadline: Sunday 2026-11-08.
Purpose: lets a new chat continue exactly where this one stopped. Read this file first, then `implementation.md` and `todo.md`. Where this file disagrees with those two, this file is newer and wins (see section 7 for the list of differences).

---

## 0. How to start the new chat

Attach or paste these three files: `handoff.md`, `implementation.md`, `todo.md`. The code lives in a public repo, so the new chat can also read it: https://github.com/dishayayyy/mlbench (branch `main`).

Suggested first message:

> I am Disha. I am continuing the MLBench project from an earlier chat. Read handoff.md, implementation.md and todo.md, check the repo, and then continue with the "Next steps" in section 6 of handoff.md. Ask me for the output of `git log --oneline -8` and `git status` first so we know the exact state.

---

## 1. Working agreement (how Disha wants to be helped)

- Start every reply with her name: "Disha".
- No emojis.
- Ask follow-up questions at the end of replies to understand needs better.
- Visual artifacts are welcome, but ask her preference before making one. She wanted only an architecture diagram and a checklist as visuals; everything else as text.
- Guide step by step. One chunk at a time, with a check command at the end of each chunk. She pastes the terminal output back and the next chunk only starts once it is confirmed. She prefers "one at a time" for code files.
- All code goes directly in chat (not only in files). Code is written like a software developer would write it: short first-person comments that read as notes to self ("note to self: ...").
- She is new to PySpark and MLflow and needs real guidance, not assumed knowledge. She is comfortable with PyTorch and Git concepts but is on Windows, so give PowerShell commands.
- Quote paths that contain her username (it has a space): `C:\Users\Disha Sharma\...`.
- She works in VS Code. The terminal must show `(.venv)` in the prompt before running project commands.
- Priority: a polished GitHub repo first, a concise technical report second. She wants genuine research-engineering rigor (experimental design, reproducibility, computational efficiency, scalability), kept manageable. She would rather have 3 models done properly than several half-done.
- Time: she can give about 5-6 focused hours per day. Today (Thursday) she planned about 2 hours from 11:30.

---

## 2. Project summary

MLBench is a reproducible benchmarking framework that evaluates models on three axes: predictive performance, computational cost, and statistical reliability.

Components: PyTorch training, repeated runs with controlled seeds, bootstrap confidence intervals, CPU/GPU benchmarking, PySpark distributed preprocessing (Pandas vs Spark comparison), MLflow experiment tracking.

Scope decisions (all agreed with Disha):
- Datasets: Adult Income, and HIGGS (1M-row training subset; full 11M rows only for the Pandas vs PySpark preprocessing benchmark). Covertype dropped.
- Models: Logistic Regression (written in PyTorch), XGBoost, MLP (PyTorch). TabNet optional, only if everything else is done. TARNet/DragonNet are a possible later extension, not part of this project.
- 10 seeds per (model, dataset), 10,000 bootstrap resamples.
- PySpark runs on Google Colab first (no Windows/WSL2 Spark setup). Local GPU and Colab GPU are benchmarked and reported separately.

Hardware (confirmed by Disha): Lenovo IdeaPad, Intel i5-12450HX, 12 GB RAM, NVIDIA RTX 2050 with 4 GB VRAM, Windows 11. NVIDIA driver 581.86 (CUDA 13.0). Colab free tier available.

---

## 3. Current environment and repo state

- Project folder: `C:\Users\Disha Sharma\projects\mlbench`
- Virtual environment: `.venv` inside the project (Python 3.12.5), selected as the VS Code interpreter. Separate from her existing conda/Jupyter environments, which are untouched.
- Key versions: torch 2.15.0+cu134 (CUDA available: True, GPU detected as RTX 2050), xgboost 3.4.1, mlflow 3.18.0, pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.4, pytest 9.2.0.
- Git identity set (name Disha Sharma, her Gmail), default branch `main`.
- Remote: https://github.com/dishayayyy/mlbench (public). First push needed a browser sign-in (worked); plain `git push` works now.
- Commits known to be pushed (oldest to newest): project skeleton; gitignore fix + planning docs; seed utility; move test to `tests/`; config + Adult loader; model interface + logistic regression; MLP + metrics + quick compare script; XGBoost + factory + configs (confirm with `git log`); "Ignore mlflow database files" (280d1ea); "Add MLflow tracking wrapper and experiment script" (1a52547, last confirmed commit).
- Tests: 19 passed at commit 1a52547.

Files that exist (all committed unless listed in section 5):

```
pyproject.toml, .gitignore, requirements-lock.txt, implementation.md, todo.md
configs/   adult_logreg.yaml, adult_mlp.yaml, adult_xgb.yaml
scripts/   quick_compare.py (scratch), run_experiment.py (first version)
src/mlbench/
  seed.py            set_seed (all libraries + deterministic flags), make_generator, seed_worker
  config.py          ExperimentConfig dataclass + load_config with validation
  metrics.py         compute_metrics -> accuracy, auc, f1, log_loss
  tracking.py        MLflow helpers: init_tracking, env_tags, flatten_params, log_config
  data/base.py       TabularSplit dataclass (X/y train, val, test, feature_names)
  data/adult.py      load_adult: OpenML fetch cached as parquet in data/, fixed stratified split
  models/base.py     BaseModel: fit(...) -> info dict, predict_proba(X) -> 1-D array
  models/torch_trainer.py  shared PyTorch binary-classifier loop (Adam, BCE, early stopping on val loss, restores best epoch, seeded generator for shuffling, chunked inference)
  models/logreg.py   one linear layer
  models/mlp.py      hidden_dims (128, 64), dropout 0.1
  models/xgb.py      XGBClassifier, hist, early stopping, subsample/colsample 0.8
  models/factory.py  build_model(cfg, seed)
tests/     test_seed, test_config, test_logreg, test_mlp, test_metrics, test_xgb, test_factory, test_tracking
```

Empty placeholder packages exist for `stats/`, `cost/`, `report/`. The `data/` folder (raw data cache) and `mlflow.db` are gitignored.

---

## 4. Key design decisions made so far (and why)

1. **Fixed data split, varying training seed.** The split uses `split_seed=123`; only the training seed changes across the 10 runs. So seed variance reflects training randomness only. The report must say this, and test-set sampling uncertainty needs a separate bootstrap over test rows.
2. **No leakage:** the scaler and one-hot encoder are fit on the training split only. Adult: `fnlwgt` dropped, missing categoricals become "Unknown". Result: 34,188 train / 4,885 val / 9,769 test rows, 107 features, positive rate 0.239 in every split.
3. **Same training loop for logreg and MLP**, so the comparison is fair. Early stopping restores the best validation epoch.
4. **XGBoost uses subsample=0.8 and colsample_bytree=0.8** on purpose. With 1.0 it would be nearly deterministic and all seeds would be identical, collapsing the confidence interval. Mention in the report.
5. **MLflow backend is a local SQLite file** (`sqlite:///mlflow.db`), not the `./mlruns` folder written in implementation.md. Layout: one experiment per dataset, a parent run per model config, a nested child run per seed. Tags record git commit (with `-dirty` if uncommitted changes), Python, OS, CPU, RAM, GPU, library versions.
6. **Seed comparisons across models are not truly paired.** Seed 3 of one model has no special link to seed 3 of another. So the model-difference bootstrap resamples each model's seeds independently. implementation.md section 5.5 says "paired comparison" and needs this correction.
7. **XGBoost on GPU:** with NumPy input, each predict call copies data CPU to GPU, so GPU timings would mostly measure the copy. Plan: keep data on the GPU in the cost profiler (Friday) so the benchmark is honest.
8. **Correction to an earlier claim:** 11M HIGGS rows x 29 float64 is about 2.5 GB, so Pandas will probably fit in 12 GB. The Pandas vs Spark benchmark must measure rather than assume that Pandas fails; Spark may well be slower at this scale, and that is a legitimate finding.
9. **HIGGS training subset is a seeded random sample** of 1M rows (decided by Disha), drawn once with a fixed sample seed that is separate from the training seeds, saved, and reused for every model and seed. The same fixed stratified split procedure as Adult is then applied.
10. **Architecture diagram:** the Mermaid version inside implementation.md is enough (decided by Disha); no rendered visual is needed.

---

## 5. Results so far (Adult, CPU, 10 seeds each, test set)

Seed 0 reference values (use them to verify reproducibility; a rerun must reproduce these exactly):

| Model | accuracy | AUC | F1 | log loss |
| --- | --- | --- | --- | --- |
| logreg | 0.8474 | 0.9046 | 0.6473 | 0.3221 |
| mlp | 0.8568 | 0.9114 | 0.6638 | 0.3157 |
| xgboost | 0.8747 | 0.9287 | 0.7112 | 0.2767 |

Range across the 10 seeds: logreg AUC 0.9046 to 0.9050 (accuracy 0.8474 to 0.8496); MLP AUC 0.9112 to 0.9128 (accuracy 0.8536 to 0.8594); XGBoost AUC 0.9278 to 0.9290 (accuracy 0.8716 to 0.8750). The ranking XGBoost > MLP > logreg holds on every seed.

These 10-seed runs are in `mlflow.db` as development runs, along with a duplicate 2-seed logreg run. They are not final. **Delete `mlflow.db` before the final runs on Saturday.**

**Code sent in chat but NOT yet confirmed as run or committed** (treat as unapplied; ask Disha for the state, then apply):
- A new `scripts/run_experiment.py` that also writes `results/<name>_<device>.csv` (one row per seed) and `results/preds/<name>_<device>.npz` (test predictions for every seed, `y_test`, seeds).
- `src/mlbench/stats/bootstrap.py` with `bootstrap_ci` (percentile CI of a statistic over values) and `bootstrap_diff_ci` (difference of means, seeds resampled independently), plus `tests/test_bootstrap.py` (4 tests; expected total 23 passed).
- `scripts/summarize.py <dataset> <device>` printing 95% bootstrap CIs per model per metric, and AUC differences between model pairs.
- After applying, rerun the three Adult configs; seed 0 must reproduce the numbers above.

---

## 6. Next steps (in order)

**Finish Thursday/Friday statistics (about 1.5 hours)**
1. Ask for `git log --oneline -8` and `git status`. Apply the unapplied code from section 5, run `pytest -q` (expect 23 passed), rerun the three Adult configs, run `python scripts/summarize.py adult cpu`, commit and push.
2. Bootstrap over test rows: use the saved prediction files to resample the 9,769 test examples and compute a CI for each model's AUC. This is the wider and more honest uncertainty. Report seed-variance and test-set-variance separately.
3. Move the loop out of `scripts/run_experiment.py` into `src/mlbench/runner.py` (the repo layout in implementation.md expects it).
4. Update `todo.md` (tick Thursday items and the finished Friday items) and fix the points in section 7.

**Friday: cost profiler and HIGGS**
5. `cost/profiler.py`: warmup iterations, `torch.cuda.synchronize()` around timed regions, training time, inference throughput, latency p50/p95, peak host memory (psutil), peak GPU memory (`torch.cuda.max_memory_allocated`). Repeat timings and report the spread. Handle XGBoost GPU data placement as in decision 7. Run CPU and local GPU for Adult.
6. HIGGS: download, convert to Parquet, loader with a 1M-row seeded random sample (decided, see decision 9), smoke run each model.

**Saturday: Spark and hardware**
7. Colab notebook: install pyspark, Spark preprocessing pipeline vs Pandas pipeline at 1M, 5M and 11M rows, partition sweep (4, 8, 16, 32), CSV vs Parquet read comparison. Save results to Drive and copy back.
8. Local GPU vs Colab GPU benchmark, reported separately; measure the speed cost of deterministic mode once.
9. Delete `mlflow.db`, commit cleanly (so the git tag is not `-dirty`), then do the final 10-seed runs for all 3 models on Adult and the HIGGS subset.

**Sunday: packaging**
10. Plots (accuracy with CIs, cost vs accuracy, Spark scaling), README with the architecture diagram, 4-6 page technical report with limitations, clean `requirements-lock.txt` (it contains a machine-specific editable-install line, and the torch line needs the CUDA index URL to reinstall), fresh-clone reproduction test, tag `v0.1`.

Cut order if time runs short: TabNet, BCa intervals, the varying-split experiment, extra Spark partition sweeps.

---

## 7. Differences between this file and implementation.md / todo.md

Update, same afternoon: `implementation.md` and `todo.md` were reissued with every correction below already applied (SQLite instead of `./mlruns`, independent seed resampling, the HIGGS random sample, the updated repo layout, ticked Thursday items). If Disha's local copies still say `./mlruns` or "paired comparison", she has the older versions and should replace them. The list is kept for reference.

- implementation.md section 5.8 and the environment table say MLflow uses `./mlruns`; it uses `sqlite:///mlflow.db`.
- implementation.md section 5.5 says paired model comparison; the actual method resamples seeds independently (decision 6).
- Repo layout additions not in implementation.md: `data/base.py`, `models/torch_trainer.py`, `models/factory.py`, `scripts/quick_compare.py` (scratch, can be deleted), `results/preds/`. `tests/` is at the project root. `runner.py` does not exist yet; its logic is currently in `scripts/run_experiment.py`.
- The Adult loader keeps the raw data cache at `data/adult_raw.parquet` (gitignored).
- todo.md: none of the Thursday boxes are ticked yet; all of them are done except "view the first run in the UI", which Disha started but not fully confirmed (see section 8).

---

## 8. Known quirks and open items

- **MLflow UI on Windows:** `Ctrl+C` can hang while a browser tab is connected. Fix: close the browser tab first, or close the terminal with the trash icon. Run the UI in a second terminal while experiments run in the first. A `WinError 10022` traceback at startup came from one worker process failing to bind the port; it is harmless. The UI opens on a GenAI home page; click the "Model training" toggle, then Experiments, then the experiment name. Whether `mlflow ui --workers 1` is accepted by this MLflow version is unconfirmed; the fallback is `mlflow server --backend-store-uri sqlite:///mlflow.db --workers 1`.
- Disha has not yet confirmed seeing the parent-run Tags (git_commit, gpu) in the UI.
- `git add .` prints LF/CRLF warnings on Windows; they are harmless.
- GitHub rejects password auth; browser sign-in worked.
- Open question for Disha: whether to attempt TabNet (optional, only if everything else is done). Resolved: HIGGS uses a seeded random sample, and the Mermaid diagram in implementation.md is enough.
