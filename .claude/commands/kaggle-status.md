Summarize the current state of this tabular-multiclass competition:

1. Print the `EXPERIMENTS_LOG.md` scoreboard, sorted by CV (best first). Note the competition
   metric and whether lower or higher is better.
2. Inventory `artifacts/oof/` and `artifacts/test_preds/` — which models exist, and for each its
   OOF CV from its `.meta.json`.
3. State the current best single model and the current best blend, with their CVs.
4. If any LB scores are recorded, show the CV↔LB gap and whether it's stable.
5. Flag anything broken: OOF files without a matching test_preds (or vice-versa), models trained
   on a stale `folds.parquet`, or missing `.meta.json`.
6. Recommend the single highest-value next action toward the goal (a feature idea, a diverse
   model, a blend re-run, or fixing the CV). Be specific and evidence-based.

Do not train anything — this is a read-only status snapshot.
