Add a new model to the pool. Argument: the model family / idea (e.g. "xgboost", "lgbm with
target-encoded cats", "mlp with 3 seeds", "ft-transformer").

Steps:
1. Pick the matching model agent (`kaggle-model-lgbm` / `-catboost` / `-mlp`) or, for a new
   family, base it on the closest `templates/train_*.py`.
2. Create `experiments/<family>_v<N>/train.py` — reuse the SHARED folds (`artifacts/folds.parquet`)
   and SHARED features (`artifacts/features_*.parquet`). Never re-fold. Integer-encode the target
   in `common.class_labels` order.
3. Train with per-fold early stopping; save per-fold models; predict OOF + test (mean over folds).
4. `common.save_oof("<family>_v<N>", ...)`, `common.save_test(...)`, `common.log_experiment(...)`.
5. Report the OOF CV (mean ± std) vs the current best, and whether it's diverse enough (check
   correlation to existing OOF) to help the blend.
6. If it's a keeper, re-run the blender (`kaggle-blend` skill) over the updated OOF pool.

Follow the contract in `COMPETITION_PLAYBOOK.md` §0 and §5. Do not blend inside the model script.
