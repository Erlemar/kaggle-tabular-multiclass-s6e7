"""
blend.py — combine all model OOF predictions into the best validated blend, apply the frozen
recipe to test predictions, write submission.csv.

Discovers every artifacts/oof/*.parquet (that also has test_preds), reports per-model CV and the
correlation matrix, hill-climbs a weighted probability average on OOF, optionally tries a
regularized stack, and keeps whichever wins on OOF. Weights come from OOF only — never the LB.

    python templates/blend.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

TRY_STACK = False
HILL_STEPS = 5
HILL_PATIENCE = 12


def hill_climb(oofs, y, lower_better):
    """Caruana greedy ensemble selection with replacement -> per-model weights."""
    k = len(oofs)
    # seed with the single best model
    singles = [C.score(y, p) for p in oofs]
    best_i = int(np.argmin(singles) if lower_better else np.argmax(singles))
    chosen = [best_i]
    cur = oofs[best_i].copy()
    best = singles[best_i]
    bad = 0
    for _ in range(HILL_STEPS):
        cand_scores = []
        for i in range(k):
            blended = (cur + oofs[i]) / (len(chosen) + 1)
            cand_scores.append(C.score(y, blended))
        i = int(np.argmin(cand_scores) if lower_better else np.argmax(cand_scores))
        improved = (cand_scores[i] < best) if lower_better else (cand_scores[i] > best)
        chosen.append(i)
        cur = cur + oofs[i]
        if improved:
            best = cand_scores[i]
            bad = 0
        else:
            bad += 1
            if bad >= HILL_PATIENCE:
                break
    w = np.bincount(chosen, minlength=k).astype(float)
    w /= w.sum()
    return w, C.score(y, sum(wi * p for wi, p in zip(w, oofs)))


def try_stack(oofs, y, folds, ncls, lower_better):
    """Multinomial logistic regression stack, OOF-validated on the same folds."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None, None
    X = np.hstack(oofs)
    stack_oof = np.zeros((len(y), ncls))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X[tr], y[tr])
        proba = lr.predict_proba(X[va])
        # align columns to full class set (LogisticRegression drops absent classes)
        full = np.zeros((va.sum(), ncls))
        full[:, lr.classes_] = proba
        stack_oof[va] = full
    return C.score(y, stack_oof), (X, y)


def main():
    ids, y, labels, oof_map = C.blend_frame()
    test_ids, test_map = C.test_frame(labels)
    names = sorted(set(oof_map) & set(test_map))
    if not names:
        print("no models with BOTH oof and test_preds — train some models first.")
        return
    folds = C.fold_array(C.load_train())
    ncls = len(labels)
    lower = not C.greater_is_better()
    oofs = [oof_map[n] for n in names]
    tests = [test_map[n] for n in names]

    print(f"metric={C.METRIC}  ({'lower' if lower else 'higher'} is better)  |  {len(names)} models\n")
    print("per-model OOF:")
    for n, p in zip(names, oofs):
        print(f"  {n:28s} {C.score(y, p):.5f}")

    # correlation of flattened OOF probabilities
    flat = np.vstack([p.ravel() for p in oofs])
    corr = np.corrcoef(flat)
    print("\ncorrelation matrix:")
    print(pd.DataFrame(corr, index=names, columns=[n[:10] for n in names]).round(3).to_string())

    # 1) hill-climbed weighted average
    w, blend_cv = hill_climb(oofs, y, lower)
    print("\nhill-climb weights:", {n: round(float(wi), 3) for n, wi in zip(names, w) if wi > 0})
    print(f"hill-climb blend OOF {C.METRIC}: {blend_cv:.5f}")

    best_kind, best_cv = "hillclimb", blend_cv
    test_blend = sum(wi * t for wi, t in zip(w, tests))

    # 2) optional regularized stack
    if TRY_STACK and len(names) >= 2:
        stack_cv, fitted = try_stack(oofs, y, folds, ncls, lower)
        if stack_cv is not None:
            print(f"stack (multinomial LR) OOF {C.METRIC}: {stack_cv:.5f}")
            better = (stack_cv < best_cv) if lower else (stack_cv > best_cv)
            # require the stack to beat the plain blend clearly (guard against overfitting)
            per_fold_std = C.cv_report(y, sum(wi * p for wi, p in zip(w, oofs)), folds)["std"]
            if better and abs(stack_cv - blend_cv) > 0.25 * per_fold_std:
                from sklearn.linear_model import LogisticRegression
                X, yy = fitted
                lr = LogisticRegression(max_iter=1000, C=1.0).fit(X, yy)
                proba = lr.predict_proba(np.hstack(tests))
                full = np.zeros((len(test_ids), ncls)); full[:, lr.classes_] = proba
                test_blend, best_kind, best_cv = full, "stack", stack_cv

    print(f"\nCHOSEN: {best_kind}  OOF {C.METRIC}={best_cv:.5f}  "
          f"(best single {min(C.score(y,p) for p in oofs) if lower else max(C.score(y,p) for p in oofs):.5f})")

    # normalize + write submission
    test_blend = np.clip(test_blend, 1e-15, 1 - 1e-15)
    test_blend = test_blend / test_blend.sum(1, keepdims=True)
    
    # 3) Metric-aware post-processing (accuracy threshold tuning)
    if C.METRIC in ("accuracy", "balanced_accuracy"):
        print(f"\noptimizing class multipliers for {C.METRIC}...")
        from scipy.optimize import minimize
        oof_blend = sum(wi * p for wi, p in zip(w, oofs))
        if best_kind == "stack":
            stack_cv, _ = try_stack(oofs, y, folds, ncls, lower)
            # Reconstruct stack OOF for tuning
            X = np.hstack(oofs)
            stack_oof = np.zeros((len(y), ncls))
            from sklearn.linear_model import LogisticRegression
            for f in sorted(np.unique(folds)):
                tr, va = folds != f, folds == f
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X[tr], y[tr])
                proba = lr.predict_proba(X[va])
                full = np.zeros((va.sum(), ncls))
                full[:, lr.classes_] = proba
                stack_oof[va] = full
            oof_blend = stack_oof

        def loss(weights):
            preds = (oof_blend * weights).argmax(axis=1)
            if C.METRIC == "balanced_accuracy":
                from sklearn.metrics import balanced_accuracy_score
                return -balanced_accuracy_score(y, preds)
            return -(preds == y).mean()
            
        best_weights = np.ones(ncls)
        best_acc = loss(best_weights)
        for _ in range(5):
            init_w = np.random.uniform(0.8, 1.2, ncls)
            res = minimize(loss, init_w, method='Nelder-Mead')
            if res.fun < best_acc:
                best_acc = res.fun
                best_weights = res.x
        
        best_weights = best_weights / best_weights.sum()
        new_cv = -best_acc
        print(f"accuracy before multipliers: {best_cv:.5f}")
        print(f"accuracy after multipliers : {new_cv:.5f}")
        print(f"best multipliers: {np.round(best_weights, 3)}")
        
        if new_cv > best_cv:
            best_cv = new_cv
            best_kind += "+tuned_mult"
            test_blend = test_blend * best_weights
            test_blend = test_blend / test_blend.sum(1, keepdims=True)

    path = C.write_submission(test_ids, test_blend, labels, path="submission_blend_v1.csv")
    print(f"wrote {path}")

    # report + log
    rep = C.ROOT / "reports" / "BLEND.md"
    rep.parent.mkdir(exist_ok=True)
    lines = [f"# Blend report ({C.METRIC})", "",
             f"- chosen: **{best_kind}**, OOF {C.METRIC} = **{best_cv:.5f}**",
             f"- members: {names}",
             f"- hill-climb weights: {[round(float(x),3) for x in w]}", "",
             "## per-model OOF", ""]
    lines += [f"- {n}: {C.score(y, p):.5f}" for n, p in zip(names, oofs)]
    rep.write_text("\n".join(lines))
    C.log_experiment("blend", best_kind, best_cv, n_feat=len(names),
                     notes=f"members={len(names)}")


if __name__ == "__main__":
    main()
