"""
make_folds.py — create the ONE shared CV split every model uses.

Writes artifacts/folds.parquet = {id, fold}. Run this ONCE after filling config.yaml and
BEFORE any model trains. If you ever change the scheme, you must regenerate every OOF.

    python src/make_folds.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GroupKFold, KFold, TimeSeriesSplit)

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C


def build_folds(train: pd.DataFrame) -> np.ndarray:
    cfg = C.CFG
    n = int(cfg["n_folds"])
    seed = int(cfg["seed"])
    scheme = cfg["fold_scheme"]
    y = train[C.TARGET].to_numpy()
    ids = train[C.ID].to_numpy()
    fold = np.full(len(train), -1, dtype=int)

    if scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
        it = splitter.split(ids, y)
    elif scheme == "stratified_group":
        g = train[cfg["group_col"]].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        it = splitter.split(ids, y, groups=g)
    elif scheme == "group":
        g = train[cfg["group_col"]].to_numpy()
        it = GroupKFold(n_splits=n).split(ids, y, groups=g)
    elif scheme == "kfold":
        it = KFold(n_splits=n, shuffle=True, random_state=seed).split(ids)
    elif scheme == "time":
        order = np.argsort(train[cfg["time_col"]].to_numpy(), kind="stable")
        it = ((order[tr], order[va]) for tr, va in TimeSeriesSplit(n_splits=n).split(order))
    else:
        raise ValueError(f"unknown fold_scheme '{scheme}'")

    for f, (_tr, va) in enumerate(it):
        fold[va] = f
    if scheme != "time":  # TimeSeriesSplit intentionally leaves early rows unused
        assert (fold >= 0).all(), "some rows got no fold — check the scheme/inputs"
    return fold


def main():
    train = C.load_train()
    fold = build_folds(train)
    out = pd.DataFrame({C.ID: train[C.ID].to_numpy(), "fold": fold})
    (C.ART / "folds.parquet").parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.ART / "folds.parquet", index=False)

    print(f"wrote {C.ART / 'folds.parquet'}  ({len(out)} rows, scheme={C.CFG['fold_scheme']}, "
          f"n_folds={C.CFG['n_folds']})")
    # class balance per fold — every fold should mirror the global distribution (except group/time)
    used = out[out["fold"] >= 0]
    tab = (pd.crosstab(used["fold"], train.loc[used.index, C.TARGET], normalize="index")
             .round(3))
    print("\nclass proportion per fold:")
    print(tab.to_string())
    print("\nrows per fold:")
    print(used["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
