---
name: kaggle-new-competition
description: Scaffold a new tabular-multiclass competition in this kit — download/place data, detect id/target/classes/metric, fill src/config.yaml, pick the fold scheme, and freeze the shared CV split. Use once at the very start, before EDA.
---

# Skill: set up a new competition

Run this once when starting a fresh competition. It gets the contract in place so every
downstream agent composes.

## Steps

1. **Get the data into `data/`.** Either place `train`, `test`, `sample_submission` there
   manually, or (if the Kaggle CLI is configured):
   ```bash
   kaggle competitions download -c <competition-slug> -p data/ && (cd data && unzip -o '*.zip')
   ```

2. **Inspect the shapes and the target.** Load `train`/`test`/`sample_submission`, print:
   dtypes, the id column (matches `sample_submission`), the target column, the **unique
   classes** and their counts, and the **prob-column layout** of `sample_submission` (this is
   the canonical class order your submission must match).

3. **Identify the metric.** Read the competition's evaluation page. Map it to one of:
   `logloss | accuracy | macro_f1 | qwk` (extend `src/metrics.py` if it's something else).
   Getting this wrong wastes the whole effort — confirm it literally.

4. **Pick the fold scheme.** Independent rows → `stratified`. Multiple rows per entity (a
   `group_col` exists and must not split across train/val) → `stratified_group`. A time column
   with test in the future → `time`. Decide from the data description, not a guess.

5. **Fill `src/config.yaml`** — `competition, id_col, target_col, metric, n_folds, fold_scheme,
   group_col, seed`, and the `data_dir`.

6. **Freeze the split and baseline:**
   ```bash
   python src/make_folds.py         # writes artifacts/folds.parquet — the shared CV
   ```
   Confirm the printed per-fold class balance looks right. Note the class-prior baseline score
   in `EXPERIMENTS_LOG.md` as the floor to beat.

7. **Hand off to EDA** (`kaggle-eda` agent / prompt 1, Stage 1).

## Guardrails
- Do NOT start modelling before `folds.parquet` exists — the split is the contract.
- Confirm `sample_submission` column order = your canonical class order; a silent permutation
  destroys logloss even with a perfect model.
