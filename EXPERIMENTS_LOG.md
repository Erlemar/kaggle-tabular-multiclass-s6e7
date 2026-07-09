# Experiments Log

The scoreboard. Every run appends a row (auto, via `common.log_experiment`). This is the source
of truth for "what we've tried and what worked." Sort mentally by CV; remember whether the metric
is lower- or higher-is-better (see `src/config.yaml`).

**Columns:** date · name · family · CV (mean±std across the shared folds) · LB (when submitted) ·
n_feat · notes.

**When you submit,** hand-edit the LB cell for that row and note the CV↔LB gap below.

<!-- log_experiment appends the table + rows here -->

## CV ↔ LB gap tracking

| submission | CV | public LB | gap | private LB | notes |
|------------|----|-----------|-----|------------|-------|
| _baseline_ | 0.85867 |           |     |            | class prior |
| blend_v1   | 0.96715 | 0.87422   | -0.09293 |       | Massive CV-LB gap! Leak or drift? |

- A **stable** gap (CV and LB move together) ⇒ trust CV to rank ideas.
- A **shifting/inverted** gap ⇒ the CV split is wrong (group/time leak, or train/test drift).
  Stop tuning and fix the split first.

## Dead ends (don't re-run)

Record anything proven not to help, with the experiment name that proved it — so no agent repeats it.

- _(none yet)_
| date | name | family | CV (mean±std) | LB | n_feat | notes |
|------|------|--------|---------------|----|--------|-------|
| 2026-07-08 18:31:25 | baseline_prior | prior | 0.85867±0.00000 |  | 0 | Class prior baseline (at-risk) |
| 2026-07-08 20:25:53 | lgbm_v1 | lightgbm | 0.96705±0.00017 |  | 140 | 1 seed(s), 6 cat |
| 2026-07-08 21:09:37 | mlp_v1 | mlp | 0.96604±0.00026 |  | 140 | emb+MLP[256, 128], 1 seed(s) |
| 2026-07-08 23:19:23 | catboost_v1 | catboost | 0.96697±0.00022 |  | 140 | 1 seed(s), 6 native-cat |
| 2026-07-08 23:19:58 | catboost_v1 | catboost | 0.96697±0.00022 |  | 140 | 1 seed(s), 6 native-cat |
| 2026-07-09 04:36:43 | blend | hillclimb+tuned_mult | 0.96715 |  | 3 | members=3 |
| 2026-07-09 04:55:34 | lgbm_v1 | lightgbm | 0.94861±0.04046 |  | 140 | 1 seed(s), 6 cat |
| 2026-07-09 05:16:11 | lgbm_v1 | lightgbm | 0.94861±0.04046 |  | 140 | 1 seed(s), 6 cat |
| 2026-07-09 06:44:54 | mlp_v1 | mlp | 0.92338±0.06135 |  | 140 | emb+MLP[256, 128], 1 seed(s) |
| 2026-07-09 06:54:30 | catboost_v1 | catboost | 0.94850±0.04041 |  | 140 | 1 seed(s), 6 native-cat |
| 2026-07-09 06:55:14 | blend | hillclimb+tuned_mult | 0.94891 |  | 3 | members=3 |
| 2026-07-09 07:52:30 | mlp_v1 | mlp | 0.94410±0.04272 |  | 140 | emb+MLP[256, 128], 1 seed(s) |
| 2026-07-09 08:00:07 | lgbm_v1 | lightgbm | 0.94505±0.04319 |  | 140 | 1 seed(s), 6 cat |
| 2026-07-09 08:33:16 | catboost_v1 | catboost | 0.94506±0.04319 |  | 140 | 1 seed(s), 6 native-cat |
| 2026-07-09 08:58:45 | lgbm_v1 | lightgbm | 0.94992±0.04339 |  | 137 | 1 seed(s), 5 cat |
| 2026-07-09 09:10:27 | mlp_v1 | mlp | 0.94957±0.04321 |  | 137 | emb+MLP[256, 128], 1 seed(s) |
| 2026-07-09 09:15:38 | blend | hillclimb+tuned_mult | 0.96768 |  | 3 | members=3 |
| 2026-07-09 09:17:23 | catboost_v1 | catboost | 0.94995±0.04340 |  | 137 | 1 seed(s), 5 native-cat |
| 2026-07-09 09:30:55 | blend | hillclimb+tuned_mult | 0.95027 |  | 3 | members=3 |
| 2026-07-09 12:23:50 | blend | hillclimb+tuned_mult | 0.95027 |  | 3 | members=3 |
| 2026-07-09 15:36:06 | catboost_v1 | catboost | 0.97470±0.00046 |  | 137 | 1 seed(s), 5 native-cat |
| 2026-07-09 15:43:41 | mlp_v1 | mlp | 0.97339±0.00042 |  | 137 | emb+MLP[256, 128], 1 seed(s) |
| 2026-07-09 16:30:43 | lgbm_v1 | lightgbm | 0.94194±0.00233 |  | 137 | 1 seed(s), 5 cat |
| 2026-07-09 16:31:31 | blend | hillclimb+tuned_mult | 0.97498 | 0.94886 | 3 | members=3 |
