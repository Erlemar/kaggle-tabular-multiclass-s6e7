# reports/

Agent-generated documentation and machine-readable findings. Append-only in spirit — add
sections, don't destroy prior analysis.

- `eda_findings.json` — machine-readable EDA output (from `templates/eda.py`); the feature agent consumes it.
- `EDA.md` — the EDA agent's human narrative + the decisions it implies.
- `FEATURES.md` — every engineered feature and **why it is not a leak**.
- `BLEND.md` — the blend members, correlation, chosen recipe, final CV.
- `DEADENDS.md` (optional) — things proven not to help, so no agent re-runs them.
