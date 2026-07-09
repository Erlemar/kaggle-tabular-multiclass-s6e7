"""
train_catboost.py — CatBoost multiclass with native categorical handling, on the shared folds.

Copy to experiments/catboost_v<N>/train.py and adapt. Obeys the common.py contract.
Let CatBoost encode categoricals natively (pass cat_features) — do NOT also target-encode them.

    python templates/train_catboost.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

TAG = "catboost_v1"
N_SEEDS = 1

PARAMS = dict(
    loss_function="MultiClass",
    depth=7,
    learning_rate=0.05,
    l2_leaf_reg=6.0,
    iterations=6000,
    od_type="Iter",
    od_wait=150,                 # early stopping patience
    random_strength=1.0,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    task_type="CPU",             # set "GPU" if a GPU is available
    verbose=0,
    # auto_class_weights="Balanced",   # enable for macro-F1 / balanced accuracy
)


def build_design():
    """Return (Xtr, Xte, cat_cols). Categoricals kept raw (NaN -> '__nan__') for native encoding."""
    train, test = C.load_train(), C.load_test()
    ftr, fte = C.ART / "features_train.parquet", C.ART / "features_test.parquet"
    if ftr.exists() and fte.exists():
        Xtr = train[[C.ID]].merge(pd.read_parquet(ftr), on=C.ID, how="left").drop(columns=[C.ID])
        Xte = test[[C.ID]].merge(pd.read_parquet(fte), on=C.ID, how="left").drop(columns=[C.ID])
    else:
        print("!! features parquet not found — using raw columns. Build features first for a real run.")
        feat = [c for c in train.columns if c not in (C.ID, C.TARGET) and c in test.columns]
        Xtr, Xte = train[feat].copy(), test[feat].copy()
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("object").where(Xtr[c].notna(), "__nan__").astype(str)
        Xte[c] = Xte[c].astype("object").where(Xte[c].notna(), "__nan__").astype(str)
    return Xtr, Xte, cat_cols


def main():
    C.set_seed()
    train, test = C.load_train(), C.load_test()
    labels = C.class_labels(train)
    y_idx = C.encode_target(train[C.TARGET].to_numpy(), labels)
    folds = C.fold_array(train)
    Xtr, Xte, cat_cols = build_design()
    n, m, ncls = len(train), len(test), len(labels)
    print(f"{TAG}: {Xtr.shape[1]} features, {len(cat_cols)} categorical (native), {ncls} classes")

    oof = np.zeros((n, ncls))
    test_pred = np.zeros((m, ncls))
    n_folds = int(C.CFG["n_folds"])
    test_pool = Pool(Xte, cat_features=cat_cols)

    for seed in range(N_SEEDS):
        for f in range(n_folds):
            tr, va = folds != f, folds == f
            if not va.any():
                continue
            model = CatBoostClassifier(**PARAMS, random_seed=C.SEED + seed,
                                       classes_count=ncls)
            model.fit(Pool(Xtr[tr], y_idx[tr], cat_features=cat_cols),
                      eval_set=Pool(Xtr[va], y_idx[va], cat_features=cat_cols),
                      use_best_model=True)
            oof[va] += model.predict_proba(Xtr[va]) / N_SEEDS
            test_pred += model.predict_proba(test_pool) / (N_SEEDS * n_folds)
            model.save_model(str(C.ART / "models" / f"{TAG}_s{seed}_f{f}.cbm"))

    rep = C.cv_report(y_idx, oof, folds)
    print(f"{TAG} OOF {C.METRIC}: {rep['overall']:.5f}  (folds {rep['mean']:.5f}±{rep['std']:.5f})")

    C.save_oof(TAG, train[C.ID].to_numpy(), oof, labels,
               meta={"family": "catboost", "n_features": int(Xtr.shape[1]), "params": PARAMS,
                     "cat_features": cat_cols, "n_seeds": N_SEEDS, "cv": rep})
    C.save_test(TAG, test[C.ID].to_numpy(), test_pred, labels)
    C.log_experiment(TAG, "catboost", rep, n_feat=Xtr.shape[1],
                     notes=f"{N_SEEDS} seed(s), {len(cat_cols)} native-cat")


if __name__ == "__main__":
    main()
