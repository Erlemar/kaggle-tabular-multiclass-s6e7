# Competition Playbook — Tabular Multiclass

The operating manual for this kit. Every agent reads this. It is opinionated on purpose:
these are the habits that separate a reproducible top-tier tabular pipeline from a pile of
notebooks that can't be combined.

> **Prime directive:** Trust cross-validation, not the public leaderboard. Build one CV split
> you believe in, make every model share it, select final submissions by CV, and treat the
> public LB as a noisy secondary signal you *watch* but do not *optimize*.

---

## 0. The contract (non-negotiable)

Multi-agent work composes **only** if everyone shares three things. These are enforced by
`src/`. Do not route around them.

1. **One CV split.** `artifacts/folds.parquet` = `{id, fold}`, created once by
   `src/make_folds.py`. Every model trains on it. Never let an agent invent its own folds.
2. **One prediction schema.** Every model writes:
   - `artifacts/oof/<name>.parquet` — one row per **train** id, columns `id, pred_<class>…`
     (out-of-fold predicted probabilities, summing to 1).
   - `artifacts/test_preds/<name>.parquet` — one row per **test** id, same prob columns
     (mean over folds).
   - `artifacts/oof/<name>.meta.json` — cv score, features, params, seed, family.
   Use `common.save_oof(...)` / `common.save_test(...)` — do not hand-roll the format.
3. **One scoreboard.** Every run appends a row to `EXPERIMENTS_LOG.md` via
   `common.log_experiment(...)`: name, family, CV (mean±std), LB (when known), n_features, notes.

If those three hold, the blender can discover and align *anything* by id. If any one breaks,
nothing downstream works.

---

## 1. Understand the problem before touching a model

- **Read the metric first, and read it literally.** Multiclass metrics reward different
  behaviour. Set `metric:` in `config.yaml` to match *exactly* and optimize *that*:
  - **Multiclass log loss / cross-entropy** → you need well-**calibrated probabilities**.
    Clip to `[1e-15, 1-1e-15]`. Blending on probabilities helps. Temperature-scaling / class-
    prior correction can pay off.
  - **Accuracy / micro-F1** → only the argmax matters. Calibration is irrelevant; class
    balance of the *decision* matters. Consider threshold/prior tuning of the argmax.
  - **Macro-F1 / balanced accuracy** → rare classes count as much as common ones. Use class
    weights, and tune per-class decision thresholds on OOF.
  - **Quadratic Weighted Kappa (ordinal)** → classes are ordered; nearby mistakes cost less.
    Regress-then-round with OOF-optimized cut points often beats plain classification.
  - **mAP@k / top-k** → rank quality of the probability vector matters, not calibration.
- **Read the data description** for the meaning of columns, the sampling unit, and any hint of
  **groups** (multiple rows per entity/customer/session/well) or **time**. This decides the
  fold scheme (§3). Getting the sampling unit wrong is the #1 cause of a CV that lies.
- **Look at `sample_submission.csv`** — it defines the exact output columns, their order, and
  the id set you must cover. Your submission must match it byte-for-byte in structure.
- **Know the class count and balance.** Severe imbalance changes everything downstream
  (stratification, class weights, metric behaviour, whether rare classes are even learnable).

## 2. EDA — what to actually look for

Run `templates/eda.py` for the machine-readable baseline, then go deeper. The findings that
change decisions:

- **Target distribution.** Counts per class; imbalance ratio; classes so rare they may be
  unlearnable (note them — they dominate macro metrics).
- **Missingness.** Per-column missing rate; whether missingness is *informative* (correlates
  with target) → add missing-indicator features. Whether train and test miss the same way.
- **Cardinality & dtype of categoricals.** Low (one-hot ok), high (target/count encoding,
  or let CatBoost handle natively), or ID-like (usually drop or hash).
- **Numeric distributions.** Skew (log1p candidates), outliers, constant / near-constant
  columns (drop), quantization/rounding (hints at how the data was made).
- **Duplicates & near-duplicates.** Exact dup rows; dup rows with *conflicting* labels (label
  noise — quantify it, it caps your achievable score).
- **Leakage suspects.** Any single feature almost-perfectly predicting the target; an `id`
  or row-order that encodes the target; columns present in train but absent/different in test.
- **Train↔test drift (adversarial validation).** Train a classifier to tell train from test.
  AUC ≈ 0.5 → same distribution, CV will track LB. AUC ≫ 0.5 → shift; find the drifting
  features, and consider adversarial-weighting or dropping unstable features. This one check
  tells you *in advance* whether your CV will correlate with the LB.
- **Feature↔feature structure.** Highly correlated clusters (redundancy), obvious ratios/
  interactions the domain suggests.

Output of EDA is a **decision list**, not a gallery: recommended fold scheme, columns to drop,
encodings to use, features to engineer, drift risks, and the label-noise ceiling.

## 3. Cross-validation — the foundation

This is where competitions are won or lost. `src/make_folds.py` supports:

- `stratified` — **default for multiclass.** Preserves class proportions per fold
  (`StratifiedKFold`). Use when rows are independent.
- `stratified_group` — rows cluster into groups (same customer/session/entity) **and** you
  want class balance. `StratifiedGroupKFold` on `group_col`. Use when the same entity must
  never be split across train and val (otherwise you leak entity identity).
- `group` — grouped, class balance not critical (`GroupKFold`).
- `time` — temporal order matters; validate on the future, train on the past
  (`TimeSeriesSplit`-style forward folds). Use when there's a time column and test is later.

Rules:
- **Same split for every model, every seed of the folding fixed.** That's the contract.
- **Report mean ± std across folds.** A model that wins on fold 0 but has high variance is
  fragile; a small mean gain inside the fold-to-fold noise is not real.
- **Track the CV↔LB gap.** Submit an early baseline to measure it. A *stable* gap (CV and LB
  move together) means you can trust CV to rank ideas. A *shifting* gap means your CV is
  wrong — fix the split (usually a grouping/time leak) before tuning anything.
- **Prefer more folds when data is small** (5 is the default; 10 for small/noisy data). Repeat
  with a second fold-seed for the *final* models to shrink selection noise — but keep OOF
  alignment: pick ONE canonical split for cross-model blending.

## 4. Feature engineering — powerful, leak-free

The rule that governs all of it: **any feature that uses the target must be computed
out-of-fold.** Compute it on the training part of each fold and apply to the held-out part,
using the *shared* `folds.parquet`. Everything else is fair game.

- **Categorical encoding**
  - *Low cardinality* → one-hot, or leave raw for CatBoost.
  - *High cardinality* → **out-of-fold target/mean encoding** (with smoothing/noise), and/or
    frequency/count encoding (no target, no leak). CatBoost's ordered boosting handles this
    natively — often just pass `cat_features`.
  - *Never* fit a target encoder on the whole training set and then score the same rows — that
    is the classic leak that inflates CV and evaporates on LB.
- **Numeric** → ratios and differences the domain suggests, group aggregations (mean/std/min/
  max/count by a categorical), rank/quantile transforms, binning, log1p for skew,
  polynomial/interaction terms for the top features.
- **Missingness** → per-column missing indicators when missingness is informative; row-wise
  missing count.
- **Model-family fit** — GBMs (LGBM/CatBoost/XGB) are invariant to monotone transforms and
  handle raw scale + NaNs; don't waste effort scaling for them. The **MLP** needs
  standardized numerics, imputed NaNs, and categorical **embeddings** (or one-hot). Build the
  shared feature frame model-agnostic; let each model apply its own last-mile prep.
- **Selection** — add features because they help **OOF CV**, not because they look clever.
  Drop features that only help one fold (fragile). Keep a `reports/FEATURES.md` describing
  every feature and *why it's not a leak*.
- **Test-aware but target-free stats are usually OK** (e.g. global frequency encoding computed
  on train+test) — but be conservative and, if unsure, compute on train only. When in doubt,
  the safe choice is: *no target information crosses a fold boundary; no test information that
  wouldn't exist at inference time.*

## 5. Models — diverse, well-validated, saved

Train **2-3 families in parallel**; diversity is what makes the blend beat the best single
model. Each model agent owns one family and must:

1. Use the shared folds + shared features.
2. Integer-encode the target to `0…C-1` in the **canonical class order** (`common.class_labels`).
3. Loop folds: fit on train part with **early stopping on the val fold**, predict val → OOF,
   predict test → accumulate. Save the per-fold model.
4. Average test predictions across folds. Score OOF with the competition metric.
5. `save_oof` + `save_test` + `log_experiment`. Never blend — that's the blender's job.

Family notes:
- **LightGBM** — fast baseline and workhorse. `objective="multiclass"`, tune
  `num_leaves`, `learning_rate` + `n_estimators` (via early stopping), `feature_fraction`,
  `bagging_fraction`, `min_child_samples`, `lambda_l1/l2`. Set `class_weight="balanced"` for
  macro metrics.
- **CatBoost** — best out-of-the-box categorical handling; pass `cat_features` and skip manual
  encoding. `loss_function="MultiClass"`, `use_best_model=True`. Strong and diverse vs LGBM.
- **XGBoost** — optional third GBM for diversity (`objective="multi:softprob"`).
- **MLP (PyTorch)** — categorical embeddings + standardized numerics; the most *decorrelated*
  member (different inductive bias) so it lifts the blend even when its solo CV trails the
  GBMs. Standardize on the train fold only; early-stop on val metric; softmax probs. Consider
  a modern tabular net (TabM / FT-Transformer / SAINT) if the MLP is competitive.
- **Seeds** — for the final models, average 3-5 seeds (seed-bagging) to cut variance. Keep OOF
  aligned to the canonical fold split.
- **Hyperparameter tuning** — start with sane defaults and a baseline. Tune with Optuna
  **inside the CV** (never tune on a single fold or the LB). Diminishing returns hit fast;
  feature engineering and blending usually beat the 4th hour of tuning.

## 6. Blending & stacking — where the last points come from

The blender is *probability-space* by default. Order of preference (simplest first):

1. **Weighted average of OOF probabilities**, weights tuned on OOF by **hill-climbing**
   (greedy add-with-replacement) or constrained optimization. Simple, robust, hard to overfit.
2. **Rank / geometric-mean averaging** — sometimes better for logloss when models are
   miscalibrated in the same direction.
3. **Stacking** — a meta-model (multinomial logistic regression, or a shallow GBM) trained on
   the OOF probability matrix, validated with the **same folds**. More powerful, easier to
   overfit — regularize hard and only keep it if it beats the plain blend on OOF *and* the gap
   is within fold noise.

Discipline:
- **Blend on OOF, apply the identical recipe to `test_preds`.** The weights/meta-model learned
  on OOF are frozen and applied to the test-prediction matrix — same models, same order.
- **Check the correlation matrix.** Blending two models correlated at 0.99 buys nothing; a
  decorrelated MLP at 0.85 can lift the blend meaningfully even with a worse solo score.
- **Don't over-blend.** More members is not better; add a member only if it improves OOF CV.
- **Metric-aware post-processing** as the final step: calibrate for logloss; optimize argmax
  thresholds / class priors for accuracy or macro-F1; round with OOF-optimized cut points for
  QWK. Always verify the post-processing helps on OOF before applying to test.

## 7. Submission — check before you spend a slot

Submissions per day are scarce. Never waste one on a formatting bug. `templates/submit.py`
(and the `kaggle-submit-check` skill) validate before you submit:

- Columns and their **order** match `sample_submission.csv` exactly.
- Every required test id is present, exactly once; no extras.
- No NaN/inf; probabilities in `[0,1]`; each row's probabilities sum to ~1 (for prob output).
- Class-column order matches the canonical order (a silent column permutation destroys logloss).
- Apply the metric's required post-processing (clip / snap / argmax→label).

**Which submissions to select as final:** pick by **CV**, then diversify. A standard safe pair
is *(a)* your best-CV blend and *(b)* a robust single-family or lower-variance blend — so a CV↔
private-LB surprise on one is hedged by the other. Do **not** pick your two highest *public-LB*
entries; that's the overfitting trap.

---

## DO / DON'T (print this)

**DO**
- Freeze one CV split; every model uses it; report mean ± std.
- Trust CV; submit an early baseline to measure the CV↔LB gap; fix CV before tuning if it drifts.
- Save OOF + test preds for **every** model in the standard schema — that's the currency of blending.
- Establish a dumb baseline (class priors), then one default GBM, before engineering anything.
- Compute all target-based features **out-of-fold**.
- Run adversarial validation early; watch for train/test drift.
- Set seeds; seed-bag the final models; log every experiment the moment it finishes.
- Build **diverse** models (different families / features / seeds) for the blend.
- Optimize the actual metric (calibrate for logloss; thresholds for F1; cut-points for QWK).
- Validate the submission format against `sample_submission.csv` before every submit.

**DON'T**
- Don't optimize the public LB or pick finals by it. Public LB is a small, noisy sample.
- Don't leak: no full-data target encoding, no scaler fit before the split, no id/row-order as
  a feature, no aggregation that includes the row's own target, no test info unavailable at inference.
- Don't blend models built on **different** fold splits — OOF won't align.
- Don't tune on one fold or on the leaderboard.
- Don't over-engineer before a baseline exists; don't chase 4th-decimal CV gains with fragile features.
- Don't reorder one frame without the other — keep everything aligned by `id`.
- Don't ship a submission whose columns/order/ids don't match `sample_submission.csv`.
- Don't ignore class imbalance or the rare-class behaviour of your metric.
- Don't declare a direction dead from an under-powered run — give each idea a fair, full build.

## Common failure modes (and the tell)

| Symptom | Likely cause |
|---|---|
| CV great, LB terrible | Leak (target encoding on full data; group/time leak in the split) |
| CV and LB both mediocre, huge gap to leaders | Missing a key feature / the intended signal; re-read data description |
| Blend ≈ best single model | Members too correlated; add a decorrelated family (MLP) or diverse features |
| Fold scores swing wildly | Too few folds / small data / a group not being held out together |
| Adversarial AUC ≫ 0.5 | Train/test drift; unstable features; CV will mis-rank ideas |
| Logloss worse than accuracy suggests | Uncalibrated probabilities; clip + calibrate |
| Great argmax metric, bad probabilistic metric (or vice-versa) | Optimizing the wrong objective for the scored metric |

## Tracking — how we keep everything

- **`EXPERIMENTS_LOG.md`** — the scoreboard. One row per run: `date · name · family · CV(mean±std)
  · LB · n_feat · notes`. Appended automatically by `common.log_experiment`. This is the source
  of truth for "what have we tried and what worked."
- **`reports/`** — `EDA.md`, `FEATURES.md`, `BLEND.md`, plus machine-readable `*_findings.json`
  the next agent consumes. Findings are append-only; add sections, don't rewrite history.
- **`artifacts/`** — the OOF/test/model artifacts, discoverable by name. The meta JSON next to
  each OOF records exactly how it was produced (features, params, seed) for reproducibility.
- **`experiments/<name>/`** — the actual training script for each run, so any result can be
  re-run. Name experiments `<family>_v<N>` (e.g. `lgbm_v1`, `catboost_v2`, `mlp_v1`).
- When something **regresses**, write it down (in the log row and/or a `reports/DEADENDS.md`)
  with the experiment id that proved it — so no agent re-runs a known dead end.

## Kernel-only competitions (read if applicable)

Some Kaggle comps score by **running your notebook** on a hidden test set swapped in at submit
time (you can't see or download the real test). If this is your competition:

- Build every per-row test artifact **inside** the scoring notebook from the mounted test dir;
  never rely on a pre-built test parquet or a hardcoded test-id list — the hidden ids differ.
- The shared `src/` module can't be imported by the Kaggle kernel runtime; **inline** the
  contract code you need into the notebook.
- A "push/run" is not a "submit" on many of these — the LB score requires the explicit
  *Submit to Competition* action, which re-runs on the hidden set and spends a daily slot.
- Everything else in this playbook (CV discipline, OOF blending, format checks) still applies —
  you just do the blending offline and bake the frozen recipe into the kernel.
