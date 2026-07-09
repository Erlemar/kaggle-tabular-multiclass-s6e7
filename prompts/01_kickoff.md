# PROMPT 1 — Kick off the multi-agent run

> Paste this to the orchestrating agent after you've put the competition data in `data/` and
> filled in `src/config.yaml`. It runs the full pipeline once, end to end, and produces a first
> submission. Prompt 2 then starts the improvement loop.

---

You are the **orchestrator** for a tabular **multiclass classification** Kaggle competition https://www.kaggle.com/competitions/playground-series-s6e7.
Your job is to run a disciplined multi-agent pipeline and produce a validated `submission.csv`.

**Read first, in this order, and follow them exactly:**
1. `COMPETITION_PLAYBOOK.md` — our approach, the do's/don'ts, and the metric-specific rules.
2. `README.md` — the workflow and the folder map.
3. `src/config.yaml` — the competition settings (id column, target, metric, fold scheme).

**The contract you must enforce on every agent (this is what makes blending possible):**
- One frozen CV split: `artifacts/folds.parquet`, created once by `src/make_folds.py`. Every
  model uses it. No agent invents its own folds.
- One prediction schema: every model writes OOF + test probabilities via `common.save_oof` /
  `common.save_test` (id-aligned, columns `pred_<class>`, summing to 1).
- One scoreboard: every run appends to `EXPERIMENTS_LOG.md` via `common.log_experiment`.

## Do this, in order

**Stage 0 — Setup (you, the orchestrator, do this yourself):**
- Verify `data/` has `train`, `test`, and `sample_submission`. Confirm `config.yaml` matches
  the real column names, the true number of classes, and the **exact competition metric**.
- Decide the fold scheme from the data (independent rows → `stratified`; grouped entities →
  `stratified_group` with `group_col`; temporal → `time`). Set it in `config.yaml`.
- Run `python src/make_folds.py` to freeze the shared CV split. Confirm class balance per fold.
- Establish the floor: a class-prior baseline, then note it in `EXPERIMENTS_LOG.md`. This is
  what every model must beat.

**Stage 1 — EDA (spawn the `kaggle-eda` agent):**
- It profiles the data and writes `reports/EDA.md` + `reports/eda_findings.json` (machine-
  readable): dtypes, cardinalities, missingness, class balance, duplicates/label-noise,
  **leakage suspects**, **adversarial-validation drift**, and a concrete recommendation list
  (fold scheme, columns to drop, encodings, features to build). It does NOT change the contract.
- Read its findings before proceeding.

**Stage 2 — Feature engineering (spawn the `kaggle-feature-engineer` agent):**
- It reads `reports/eda_findings.json` and builds `src/features.py`, producing
  `artifacts/features_train.parquet` + `artifacts/features_test.parquet` (id-aligned).
- **All target-based features must be computed out-of-fold using `artifacts/folds.parquet`.**
  It documents every feature and why it's leak-free in `reports/FEATURES.md`.
- Sanity-check: no target leakage, train/test feature parity, ids aligned.

**Stage 3 — Models (spawn 2-3 model agents IN PARALLEL, one message, multiple tool calls):**
- `kaggle-model-lgbm`, `kaggle-model-catboost`, and `kaggle-model-mlp`.
- Each uses the shared folds + shared features, trains with per-fold early stopping, saves
  per-fold models, writes OOF + test predictions in the standard schema, and logs a row.
- Each reports its OOF CV (mean ± std). None of them blends.
- They are independent — run them concurrently.

**Stage 4 — Blend & submit (spawn the `kaggle-blender` agent):**
- It discovers all OOF via `common`, computes each model's CV and the **correlation matrix**,
  then finds the best combination on OOF (hill-climb weighted average first; try a regularized
  stack only if it beats the plain blend within fold noise).
- It applies the identical frozen recipe to the test predictions, does metric-aware post-
  processing, writes `submission.csv`, and runs the `kaggle-submit-check` skill to validate the
  format against `sample_submission.csv`. It writes `reports/BLEND.md` and logs the final CV.

## Rules for you as orchestrator
- Enforce the contract on every sub-agent; reject any output that breaks OOF alignment or the schema.
- After each stage, briefly summarize what came back and what you're doing next — don't dump raw output.
- Address every stage fully; don't skip the MLP or the adversarial-validation check because they
  "seem marginal." Diversity and drift-awareness are where the points are.
- When the pipeline finishes, report: the per-model CVs, the blend CV, the correlation matrix
  summary, the chosen recipe, and the exact `submission.csv` path — then stop and wait. I'll
  submit to the leaderboard and give you the LB score plus a goal (Prompt 2).

Begin with Stage 0 now.
