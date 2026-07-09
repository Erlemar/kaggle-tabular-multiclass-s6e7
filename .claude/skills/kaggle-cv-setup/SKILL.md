---
name: kaggle-cv-setup
description: Create or validate the shared cross-validation split (artifacts/folds.parquet) that every model must use, and check that it tracks the leaderboard. Use before any model trains, or whenever CV and LB diverge.
---

# Skill: shared CV setup & validation

The single most important artifact in the kit. Every model trains on this exact split so their
OOF predictions align and can be blended.

## Create the split
```bash
python src/make_folds.py
```
Writes `artifacts/folds.parquet` = `{id, fold}` using `fold_scheme` from `config.yaml`:
- `stratified` — default for multiclass (class-balanced folds; independent rows).
- `stratified_group` — class-balanced AND keeps each `group_col` entity in one fold (no entity leak).
- `group` — grouped, balance not critical.
- `time` — forward-chaining folds for temporal data.

## Validate the split
- **Class balance per fold** — printed by `make_folds.py`; each fold should mirror the global
  class distribution (except intentionally for `group`/`time`).
- **No entity leak** — for grouped data, confirm no `group_col` value appears in two folds.
- **CV↔LB gap** — after the first real submission, compare CV to LB. A *stable* gap (they move
  together) means CV is trustworthy. A *shifting/inverted* gap means the split is wrong.

## If CV doesn't track LB — fix the split first
Before tuning anything else, diagnose:
- Is there a hidden **group** (same entity in many rows) you're splitting across folds? → `stratified_group`.
- Is the test set **later in time**? → `time` folds.
- Is there **train/test drift** (adversarial AUC ≫ 0.5)? → the CV distribution differs from test;
  consider adversarial validation weighting or dropping unstable features.
- Is a **leak** inflating CV? → audit target-based features for out-of-fold correctness.

## Guardrails
- **One canonical split for cross-model blending.** You may repeat with a second fold-seed to
  shrink final-model variance, but pick ONE split as the alignment reference for OOF.
- Never let a model re-fold on its own. If `folds.parquet` changes, all OOF must be regenerated.
