---
name: kaggle-blend
description: Combine all model OOF predictions into the best validated blend/stack, apply the frozen recipe to test predictions, and write submission.csv. Use after the model agents finish, or any time the OOF pool changes.
---

# Skill: blend / stack the model pool

Turn the pool of per-model OOF + test predictions into one submission that beats every single
model — validated on the shared folds, never on the leaderboard.

## Run it
```bash
python templates/blend.py            # discovers all artifacts/oof/*.parquet, writes submission.csv
```

## What it does (and what to check)
1. **Discover** every `artifacts/oof/<name>.parquet` and its matching `test_preds`. Align to the
   train target by `id`.
2. **Diagnose** — print each model's OOF CV and the **pairwise correlation matrix**. Redundant
   members (corr ≈ 0.99) add nothing; diverse ones (a decorrelated MLP) lift the blend.
3. **Search on OOF**, simplest first:
   - hill-climb **weighted average** (greedy add-with-replacement) — the robust default;
   - **rank / geometric-mean** average — for similarly-miscalibrated models;
   - a **regularized stack** (multinomial LR / shallow GBM) on the OOF prob matrix with the same
     folds — keep only if it beats the plain blend on OOF beyond fold noise.
4. **Freeze & apply** — the winning weights/meta-model are frozen and applied to the **test**
   prediction matrix (same members, same order). Blend weights come from OOF, *never* from LB.
5. **Metric-aware post-processing** (last): clip for logloss; OOF-tuned argmax thresholds /
   class priors for accuracy or macro-F1; OOF cut points for QWK. Verify each helps on OOF first.
6. **Write** `submission.csv`, then run the `kaggle-submit-check` skill.

## Guardrails
- Add a member only if it improves **OOF CV**. Don't over-blend; simpler generalizes better.
- Blend on OOF, apply identically to test. If you ever fit weights on the LB, you've overfit it.
- Record the recipe + member weights + final CV in `reports/BLEND.md` and the scoreboard.
