# Kaggle Tabular Multiclass — Multi-Agent Starter Kit

A portable, self-contained kit for solving a **tabular multiclass classification** Kaggle
competition with a coordinated team of Claude Code agents:

```
EDA  →  Feature Engineering  →  {LightGBM, CatBoost, MLP} (parallel)  →  Blend/Stack  →  Submission
```

Copy this whole folder to the machine where you'll compete, drop the competition data into
`data/`, fill in `src/config.yaml`, and paste **`prompts/01_kickoff.md`** to your agent.
Then paste **`prompts/02_iterate.md`** to start the improvement loop.

---

## Why this kit exists

Multi-agent pipelines only work if every agent agrees on **one contract**. If two model
agents invent their own cross-validation splits, their out-of-fold (OOF) predictions can't
be blended, and the whole exercise collapses. This kit encodes that contract in code:

- **One frozen CV split** (`artifacts/folds.parquet`) that *every* model uses.
- **One OOF / test-prediction schema** (`src/common.py`) that *every* model writes.
- **One scoreboard** (`EXPERIMENTS_LOG.md`) that *every* agent appends to.

Get those three right and blending becomes trivial. Get them wrong and nothing composes.

## The 60-second quickstart

```bash
# 0. one-time
pip install -r requirements.txt
#    put the competition files in ./data/  (train.csv, test.csv, sample_submission.csv)
#    edit src/config.yaml  (id_col, target_col, metric, fold_scheme)

# 1. freeze the shared CV split (do this ONCE, before any model runs)
python src/make_folds.py

# 2. sanity-check EDA artifact (the EDA agent will go deeper)
python templates/eda.py

# 3+  the agents take over from prompts/01_kickoff.md
```

## The workflow (what the agents do)

| Stage | Agent | Reads | Produces |
|-------|-------|-------|----------|
| 0. Setup | orchestrator | `config.yaml` | `artifacts/folds.parquet`, a prior/baseline score |
| 1. EDA | `kaggle-eda` | raw `data/` | `reports/EDA.md`, `reports/eda_findings.json` |
| 2. Features | `kaggle-feature-engineer` | EDA findings | `src/features.py`, `artifacts/features_{train,test}.parquet`, `reports/FEATURES.md` |
| 3. Models (×2-3, parallel) | `kaggle-model-lgbm` / `-catboost` / `-mlp` | folds + features | `artifacts/oof/<name>.parquet`, `artifacts/test_preds/<name>.parquet`, models, a log row |
| 4. Blend | `kaggle-blender` | all OOF | `submission.csv`, best-blend report, final CV |
| ↺ Iterate | orchestrator | scoreboard + reports | new hypotheses → more feature/model/blend rounds |

## Folder map

```
kaggle-tabular-multiclass/
├── README.md                      ← you are here
├── COMPETITION_PLAYBOOK.md        ← the master approach (do's, don'ts, tracking) — READ THIS
├── EXPERIMENTS_LOG.md             ← the scoreboard (every run appends one row)
├── requirements.txt
├── prompts/
│   ├── 01_kickoff.md              ← paste to launch the multi-agent run
│   └── 02_iterate.md              ← paste to start the goal-driven improvement loop
├── .claude/
│   ├── agents/                    ← the cast: orchestrator, eda, feature-engineer, 3 model trainers, blender
│   ├── skills/                    ← reusable procedures (cv-setup, blend, submit-check, new-competition)
│   └── commands/                  ← slash-command entry points
├── src/                           ← the SHARED CONTRACT — imported by everything, never forked per-model
│   ├── config.yaml                ← the one place you set id/target/metric/folds
│   ├── common.py                  ← data IO, folds, OOF/test schema, scoring, experiment logging
│   ├── metrics.py                 ← multiclass metric dispatch (logloss / accuracy / macro-F1 / QWK)
│   └── make_folds.py              ← creates artifacts/folds.parquet ONCE
├── templates/                     ← copy-and-fill scripts (agents adapt these)
│   ├── eda.py  train_lgbm.py  train_catboost.py  train_mlp.py  blend.py  submit.py
├── experiments/                   ← one subdir per model run (agents create these)
├── reports/                       ← EDA.md, FEATURES.md, BLEND.md, findings JSON
├── artifacts/                     ← folds.parquet, oof/, test_preds/, models/  (git-ignored, big)
└── data/                          ← competition CSVs go here  (git-ignored)
```

## The one rule that matters most

**Trust your cross-validation, not the public leaderboard.** The public LB is scored on a
small slice of the test set; tuning to it is how people fall 500 places on the private
reveal. Build a CV you trust, track the CV↔LB gap, and *select your final submissions by CV*.
Everything in `COMPETITION_PLAYBOOK.md` flows from this.

---

*Kit is competition-agnostic. It assumes CSV-submission comps. If your competition is
**kernel-only** (submit a notebook, hidden test swapped in at scoring), read the
"Kernel-only competitions" section of the playbook before you start — the contract changes.*
