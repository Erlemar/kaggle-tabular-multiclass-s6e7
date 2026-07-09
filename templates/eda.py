"""
eda.py — machine-readable EDA baseline for tabular multiclass.

Writes reports/eda_findings.json (consumed by the feature agent) and prints a human summary.
The EDA agent runs this first, then goes deeper. Robust to arbitrary column sets.

    python templates/eda.py
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

warnings.filterwarnings("ignore")
HIGH_CARD = 50          # categorical cardinality above which we recommend target/count enc
NEAR_CONST_TOL = 0.999  # a column whose top value covers > this share is near-constant


def _adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list):
    """Train-vs-test classifier AUC + top drifting features. AUC ~0.5 => same distribution."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        return None, [], f"sklearn unavailable: {e}"
    cols = [c for c in feat_cols if c in test.columns]
    if not cols:
        return None, [], "no shared feature columns"
    X = pd.concat([train[cols], test[cols]], ignore_index=True)
    for c in cols:
        if X[c].dtype == object or str(X[c].dtype).startswith("category"):
            X[c] = pd.factorize(X[c])[0]
    X = X.apply(pd.to_numeric, errors="coerce")
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=C.SEED)
    try:
        proba = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
        clf.fit(X, y)
        imp = getattr(clf, "feature_importances_", None)
        drift = []
        if imp is not None:
            drift = [cols[i] for i in np.argsort(imp)[::-1][:10]]
        else:  # permutation-free fallback: rank by |mean train - mean test| standardized
            diffs = {c: abs(pd.to_numeric(train[c], errors="coerce").mean()
                            - pd.to_numeric(test[c], errors="coerce").mean()) / (X[c].std() + 1e-9)
                     for c in cols if pd.api.types.is_numeric_dtype(X[c])}
            drift = [k for k, _ in sorted(diffs.items(), key=lambda kv: -kv[1])[:10]]
        return auc, drift, ""
    except Exception as e:
        return None, [], f"adversarial failed: {e}"


def _leakage_suspects(train: pd.DataFrame, num_cols, y_idx, k=10):
    """Rank features by univariate mutual information with the target — a very high value on a
    single raw feature is a leak suspect worth a human look."""
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        return []
    if not num_cols:
        return []
    X = train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(train[num_cols].median())
    try:
        mi = mutual_info_classif(X, y_idx, random_state=C.SEED)
        order = np.argsort(mi)[::-1]
        return [{"feature": num_cols[i], "mutual_info": round(float(mi[i]), 4)} for i in order[:k]]
    except Exception:
        return []


def main():
    train = C.load_train()
    test = C.load_test()
    labels = C.class_labels(train)
    y = train[C.TARGET]
    y_idx = C.encode_target(y.to_numpy(), labels)

    feat_cols = [c for c in train.columns if c not in (C.ID, C.TARGET)]
    num_cols, cat_cols = [], []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # --- class balance ---
    counts = y.value_counts().sort_index()
    class_counts = {str(k): int(v) for k, v in counts.items()}
    imbalance = float(counts.max() / max(counts.min(), 1))

    # --- missingness ---
    miss = (train[feat_cols].isna().mean().sort_values(ascending=False))
    missing = {c: round(float(m), 4) for c, m in miss.items() if m > 0}
    # informative missingness: does the missing-indicator correlate with the target index?
    missing_informative = []
    for c, m in list(missing.items())[:50]:
        if 0 < m < 1:
            ind = train[c].isna().astype(int).to_numpy()
            if ind.std() > 0 and abs(np.corrcoef(ind, y_idx)[0, 1]) > 0.03:
                missing_informative.append(c)

    # --- categorical cardinality + recommended encoding ---
    cat_info = {}
    for c in cat_cols:
        card = int(train[c].nunique(dropna=True))
        if card <= 2:
            enc = "onehot_or_binary"
        elif card <= HIGH_CARD:
            enc = "onehot_or_native_catboost"
        else:
            enc = "oof_target_encode + frequency_encode (or native CatBoost)"
        cat_info[c] = {"cardinality": card, "recommended_encoding": enc}

    # --- near-constant / skew ---
    near_const, log1p_candidates = [], []
    for c in num_cols:
        s = train[c]
        top = s.value_counts(normalize=True, dropna=False)
        if len(top) and top.iloc[0] > NEAR_CONST_TOL:
            near_const.append(c)
        s2 = s.dropna()
        if len(s2) > 10 and (s2 >= 0).all() and abs(float(s2.skew())) > 2:
            log1p_candidates.append(c)

    # --- duplicates / label noise ---
    dup_rows = int(train.duplicated(subset=feat_cols).sum())
    conflict_rate = 0.0
    if dup_rows:
        g = train.groupby(feat_cols, dropna=False)[C.TARGET].nunique()
        conflicting = int((g > 1).sum())
        conflict_rate = round(conflicting / max(len(g), 1), 5)

    # --- id leakage: does the id (if numeric/ordinal) track the target? ---
    id_leak = None
    try:
        idnum = pd.to_numeric(train[C.ID], errors="coerce")
        if idnum.notna().all() and idnum.std() > 0:
            id_leak = round(abs(float(np.corrcoef(idnum, y_idx)[0, 1])), 4)
    except Exception:
        pass

    adv_auc, drift_features, adv_note = _adversarial_auc(train, test, feat_cols)
    leak_suspects = _leakage_suspects(train, num_cols, y_idx)

    # --- fold-scheme hint ---
    fold_hint = "stratified"
    if C.CFG.get("group_col"):
        fold_hint = "stratified_group"
    elif C.CFG.get("time_col"):
        fold_hint = "time"

    findings = {
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "id_col": C.ID,
        "target_col": C.TARGET,
        "metric": C.METRIC,
        "classes": [str(x) for x in labels],
        "n_classes": len(labels),
        "class_counts": class_counts,
        "imbalance_ratio": round(imbalance, 3),
        "recommended_fold_scheme": fold_hint,
        "group_col_candidate": C.CFG.get("group_col"),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "categorical_info": cat_info,
        "missing_rate": missing,
        "missing_indicator_cols": missing_informative,
        "near_constant_cols": near_const,
        "log1p_candidates": log1p_candidates,
        "duplicate_rows": dup_rows,
        "duplicate_conflict_rate": conflict_rate,
        "id_target_correlation": id_leak,
        "adversarial_auc": adv_auc,
        "drift_features": drift_features,
        "adversarial_note": adv_note,
        "leakage_suspects": leak_suspects,
        "engineered_feature_ideas": [
            "out-of-fold target/mean encoding for high-cardinality categoricals",
            "frequency/count encoding for all categoricals",
            "group aggregations (mean/std/min/max/count) of numerics by each categorical",
            "ratios/differences between correlated numeric pairs",
            "missing-indicator columns for informative-missing features",
            "rank/quantile transforms; log1p for the skew candidates",
        ],
    }

    out = C.ROOT / "reports" / "eda_findings.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str))

    # --- human summary ---
    print(f"train {train.shape}  test {test.shape}  | {len(labels)} classes  metric={C.METRIC}")
    print(f"class counts: {class_counts}  (imbalance {imbalance:.1f}x)")
    print(f"numeric: {len(num_cols)}  categorical: {len(cat_cols)}")
    if missing:
        print(f"top missing: {dict(list(missing.items())[:5])}")
    if near_const:
        print(f"near-constant (drop candidates): {near_const[:10]}")
    print(f"duplicate rows: {dup_rows}  conflicting-label rate: {conflict_rate}")
    print(f"adversarial train/test AUC: {adv_auc}  (~0.5 = no drift; >>0.5 = shift)  {adv_note}")
    if drift_features:
        print(f"top drift features: {drift_features[:8]}")
    if leak_suspects:
        print(f"leakage suspects (high MI): {[s['feature'] for s in leak_suspects[:5]]}")
    if id_leak and id_leak > 0.1:
        print(f"** id correlates with target ({id_leak}) — possible id/order leak **")
    print(f"\nwrote {out}")
    print("=> the EDA agent should now write reports/EDA.md with the decisions this implies.")


if __name__ == "__main__":
    main()
