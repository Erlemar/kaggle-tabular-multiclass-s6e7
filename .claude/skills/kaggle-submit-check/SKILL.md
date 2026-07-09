---
name: kaggle-submit-check
description: Validate a submission.csv against sample_submission before spending a scarce daily submission slot — column names/order, id coverage, prob ranges/sums, NaNs, and canonical class order. Use right before every leaderboard submission.
---

# Skill: pre-submission format check

Submissions per day are scarce. Never burn one on a formatting bug. Validate first.

## Run it
```bash
python templates/submit.py --check           # validate only
python templates/submit.py --check --submit -m "blend_v3: cv 0.4123"   # validate then submit (if Kaggle CLI set up)
```

## What must pass (all of these)
- **Columns match** `sample_submission.csv` exactly — same names, **same order**. For probability
  submissions the class-column order is the canonical order; a silent permutation destroys logloss.
- **Ids** — every required test id present, exactly once; no missing, no extras.
- **Values** — no NaN/inf; probabilities in `[0,1]`; each row's probabilities sum to ≈ 1
  (for prob submissions). For argmax/label submissions, every value is a valid class label.
- **Post-processing applied** — the metric's required transform (clip / label-map / snap) is done.
- **Row count** = `sample_submission` row count.

## Then submit (kernel vs CSV)
- **CSV competition:** `kaggle competitions submit -c <slug> -f submission.csv -m "<msg>"`, then
  check `kaggle competitions submissions -c <slug>`.
- **Kernel-only competition:** the CSV upload does not score. Push the notebook and use the UI
  *Submit to Competition* action — it re-runs on the hidden test and spends the slot. (See the
  playbook's "Kernel-only" section.)

## After submitting
- Record CV → LB in `EXPERIMENTS_LOG.md`. Update the CV↔LB gap. If CV and LB disagree, the CV is
  suspect — investigate the split/leak before trusting more experiments.
- **Select finals by CV, not public LB.** A safe pair: best-CV blend + a lower-variance hedge.
