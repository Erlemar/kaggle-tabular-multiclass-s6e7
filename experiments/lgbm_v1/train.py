"""
train_lgbm.py — LightGBM multiclass on the shared folds + features, writes OOF + test preds.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

TAG = "lgbm_v1"
N_SEEDS = 1

PARAMS = dict(
    objective="multiclass",
    metric="multi_logloss",          # early-stopping proxy; OOF is scored with the real metric
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=40,
    lambda_l1=0.0,
    lambda_l2=1.0,
    verbosity=-1,
)
NUM_BOOST_ROUND = 5000
EARLY_STOP = 150


def build_design():
    """Return (Xtr, Xte, cat_cols) aligned to train[ID] / test[ID]. Prefers the shared feature
    parquets; falls back to raw columns (numeric as-is, categoricals -> category dtype)."""
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
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    return Xtr, Xte, cat_cols


def main():
    C.set_seed()
    train, test = C.load_train(), C.load_test()
    labels = C.class_labels(train)
    y_idx = C.encode_target(train[C.TARGET].to_numpy(), labels)
    folds = C.fold_array(train)
    Xtr, Xte, cat_cols = build_design()
    n, m, ncls = len(train), len(test), len(labels)
    print(f"{TAG}: {Xtr.shape[1]} features, {len(cat_cols)} categorical, {ncls} classes")

    oof = np.zeros((n, ncls))
    test_pred = np.zeros((m, ncls))
    n_folds = int(C.CFG["n_folds"])

    for seed in range(N_SEEDS):
        params = {**PARAMS, "num_class": ncls, "seed": C.SEED + seed,
                  "feature_fraction_seed": C.SEED + seed, "bagging_seed": C.SEED + seed}
        for f in range(n_folds):
            tr, va = folds != f, folds == f
            if not va.any() or not tr.any():
                continue
            sample_weights = compute_sample_weight("balanced", y=y_idx[tr])
            dtr = lgb.Dataset(Xtr[tr], label=y_idx[tr], categorical_feature=cat_cols or "auto", weight=sample_weights)
            dva = lgb.Dataset(Xtr[va], label=y_idx[va], reference=dtr)
            booster = lgb.train(
                params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
            )
            oof[va] += booster.predict(Xtr[va]) / N_SEEDS
            test_pred += booster.predict(Xte) / (N_SEEDS * n_folds)
            
            # Ensure models directory exists
            (C.ART / "models").mkdir(parents=True, exist_ok=True)
            booster.save_model(str(C.ART / "models" / f"{TAG}_s{seed}_f{f}.txt"))

    rep = C.cv_report(y_idx, oof, folds)
    print(f"{TAG} OOF {C.METRIC}: {rep['overall']:.5f}  (folds {rep['mean']:.5f}±{rep['std']:.5f})")

    C.save_oof(TAG, train[C.ID].to_numpy(), oof, labels,
               meta={"family": "lightgbm", "n_features": int(Xtr.shape[1]), "params": PARAMS,
                     "n_seeds": N_SEEDS, "cv": rep})
    C.save_test(TAG, test[C.ID].to_numpy(), test_pred, labels)
    C.log_experiment(TAG, "lightgbm", rep, n_feat=Xtr.shape[1],
                     notes=f"{N_SEEDS} seed(s), {len(cat_cols)} cat")


if __name__ == "__main__":
    main()
