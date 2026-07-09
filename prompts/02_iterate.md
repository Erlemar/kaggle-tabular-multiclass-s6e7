# PROMPT 2 — Set a goal and iterate

> Paste this after the first pipeline run (Prompt 1) has produced a baseline submission and you
> have (optionally) an LB score. Fill in the GOAL line. This starts the closed-loop improvement
> process. Re-paste / continue it across rounds until the goal is met or ideas are exhausted.

---

**GOAL:** reach 0.93 on public LB

**Latest LB:** 0.87228

You are the orchestrator. Improve our result toward the GOAL by iterating the pipeline. Work
empirically: hypothesize from evidence, build the idea **fully**, let **CV** judge it, keep what
wins, log everything.

## The loop (repeat each round)

1. **Assess.** Read `EXPERIMENTS_LOG.md`, `reports/EDA.md`, `reports/FEATURES.md`,
   `reports/BLEND.md`, and the current best CV. Update the CV↔LB gap from the latest LB. State
   where we are vs the GOAL and what the current bottleneck is (missing signal? weak diversity?
   overfitting? a CV that doesn't track LB?).
2. **Prioritize.** Brainstorm and come up with 3-5 highest-expected-value ideas for this round, sorted by
   expected gain × probability. Bias toward what the evidence supports. Typical high-value moves:
   - **Features**: interactions/aggregations the EDA flagged; better categorical encoding;
     out-of-fold target encoding done right; drift-robust feature selection; missing-indicators.
   - **Models**: tune the GBMs with Optuna *inside CV*; add a diverse family (XGBoost, a modern
     tabular net); seed-bag the finals; class-weighting for the metric.
   - **Blend**: re-optimize weights over the new OOF; try a regularized stack; add only members
     that improve OOF CV and are decorrelated.
   - **Metric fit**: calibration for logloss; per-class threshold / prior tuning for
     accuracy/macro-F1; OOF-optimized cut points for QWK.
3. **Execute.** Spawn the relevant agent(s) — parallel when independent (multiple feature or
   model experiments at once), sequential when one depends on another. Every experiment obeys
   the contract: shared folds, standard OOF/test schema, a logged row. Name runs
   `<family>_v<N>`. Build each idea at full fidelity — no under-powered proxies, no "seems
   marginal so skip."
4. **Judge & record.** Compare on OOF CV (mean ± std). Keep gains that exceed fold-to-fold
   noise; discard the rest and note the dead end so we don't repeat it. Re-run the blender over
   the updated OOF pool and update `submission.csv` only if the blend CV improves.
5. **Decide the next round.** If the GOAL isn't met and there are promising ideas left, loop. If
   a submission slot is worth spending, tell me exactly which candidate to submit and the CV
   evidence for it — I'll submit and report the LB back so you can recalibrate.

## Guardrails (do not violate while iterating)
- **CV is the judge, not the public LB.** Never tune to the LB or pick finals by it. Track the
  gap; if CV stops predicting LB, stop and fix the CV before doing anything else.
- **Protect the contract.** New models must reuse `artifacts/folds.parquet` and the standard
  schema, or their OOF can't be blended. Never silently re-fold.
- **No leakage, ever.** Re-audit any new target-based feature for out-of-fold correctness before
  trusting its CV. A too-good CV jump is a leak until proven otherwise.
- **Diversity beats a 4th-decimal tune.** When the best single model plateaus, add a
  decorrelated member rather than grinding one family's hyperparameters.
- **Address every idea you list.** If you enumerate 5 ideas for a round, build all 5 (or
  explicitly say which you're deferring and why) — don't quietly drop the ones you guess are
  marginal; compute is not the constraint, and a weak guess shouldn't close a door.
- **Log immediately.** Every run gets a scoreboard row the moment it finishes, win or lose.

## Each round, report back
- Current best CV (and LB if newly known) vs the GOAL; the CV↔LB gap.
- What you tried this round, each result vs the prior best, and what you kept / discarded / why.
- The updated blend CV and whether `submission.csv` changed.
- The plan for the next round, or — if you recommend spending a submission slot — the exact
  candidate and the CV case for it.

Start the loop now with an Assess step.
