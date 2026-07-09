"""
common.py — the shared contract for the multi-agent pipeline.

Import this from every EDA / feature / model / blend script:

    import common as C

so that all agents agree on:
  * the data location and the target/id columns          (config.yaml)
  * the frozen CV split                                  (artifacts/folds.parquet)
  * the OOF / test-prediction schema                     (save_oof / save_test)
  * the canonical class order                            (class_labels)
  * the competition metric                               (score)
  * the experiment scoreboard                            (log_experiment)

Do NOT fork or copy this file per model — one source of truth.

Import boilerplate (robust to being run from anywhere / a copied experiment subdir):

    import sys
    from pathlib import Path
    _p = Path(__file__).resolve()
    for _a in [_p, *_p.parents]:
        if (_a / "src" / "common.py").exists():
            sys.path.insert(0, str(_a / "src")); break
    import common as C
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import metrics as _metrics


# --------------------------------------------------------------------------------------
# Locate the kit root (the folder that contains src/common.py) and load config
# --------------------------------------------------------------------------------------
def _find_root(start: Path) -> Path:
    for anc in [start, *start.parents]:
        if (anc / "src" / "common.py").exists():
            return anc
    return start


ROOT = _find_root(Path(__file__).resolve())


def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))
    for key in ("data_dir", "artifacts_dir"):
        p = cfg[key]
        cfg[key] = p if os.path.isabs(p) else str((ROOT / p).resolve())
    cfg["_root"] = str(ROOT)
    return cfg


CFG = load_config()
DATA = Path(CFG["data_dir"])
ART = Path(CFG["artifacts_dir"])
for _sub in ("oof", "test_preds", "models"):
    (ART / _sub).mkdir(parents=True, exist_ok=True)

ID = CFG["id_col"]
TARGET = CFG["target_col"]
METRIC = CFG["metric"]
SEED = int(CFG["seed"])


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int | None = None) -> int:
    seed = SEED if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


# --------------------------------------------------------------------------------------
# Data IO (csv or parquet, auto-detected)
# --------------------------------------------------------------------------------------
def _read_any(stem: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        p = DATA / f"{stem}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"no {stem}.parquet/.csv in {DATA}")


def load_train() -> pd.DataFrame:
    df = _read_any("train")
    
    clean_path = ART / "clean_ids.parquet"
    if clean_path.exists():
        clean_ids = pd.read_parquet(clean_path)[ID].tolist()
        df = df[df[ID].isin(clean_ids)].reset_index(drop=True)
        
    return df


def load_test() -> pd.DataFrame:
    return _read_any("test")


def load_sample_submission() -> pd.DataFrame:
    for name in ("sample_submission", "sampleSubmission"):
        for ext in (".csv", ".parquet"):
            p = DATA / f"{name}{ext}"
            if p.exists():
                return pd.read_csv(p) if ext == ".csv" else pd.read_parquet(p)
    raise FileNotFoundError(f"no sample_submission.csv in {DATA}")


# --------------------------------------------------------------------------------------
# Classes: the canonical label order used for every probability matrix and column
# --------------------------------------------------------------------------------------
def class_labels(train: pd.DataFrame | None = None) -> list:
    """Canonical class order. Uses config `classes` if set, else sorted-unique of the target."""
    if CFG.get("classes"):
        return list(CFG["classes"])
    if train is None:
        train = load_train()
    return sorted(train[TARGET].dropna().unique().tolist())


def encode_target(y, labels: list) -> np.ndarray:
    """Map raw labels -> integer index (0..C-1) in canonical order."""
    lut = {lab: i for i, lab in enumerate(labels)}
    return np.asarray([lut[v] for v in y], dtype=np.int64)


def decode_target(idx, labels: list):
    labels = list(labels)
    return np.asarray([labels[int(i)] for i in idx])


def prob_cols(labels: list) -> list:
    return [f"pred_{lab}" for lab in labels]


# --------------------------------------------------------------------------------------
# The shared CV split
# --------------------------------------------------------------------------------------
def load_folds() -> pd.DataFrame:
    p = ART / "folds.parquet"
    if not p.exists():
        raise FileNotFoundError(
            "artifacts/folds.parquet missing — run `python src/make_folds.py` first. "
            "The shared split is the contract; never let a model make its own."
        )
    return pd.read_parquet(p)


def fold_array(train: pd.DataFrame) -> np.ndarray:
    """Fold id per row of `train`, aligned by ID (order-safe)."""
    folds = load_folds().set_index(ID)["fold"]
    return train[ID].map(folds).to_numpy()


# --------------------------------------------------------------------------------------
# OOF / test prediction schema  (the currency of blending)
# --------------------------------------------------------------------------------------
def save_oof(name: str, ids, probs: np.ndarray, labels: list, meta: dict | None = None) -> Path:
    probs = np.asarray(probs, dtype=np.float64)
    assert probs.shape[1] == len(labels), "prob columns != n classes"
    df = pd.DataFrame(probs, columns=prob_cols(labels))
    df.insert(0, ID, np.asarray(ids))
    out = ART / "oof" / f"{name}.parquet"
    df.to_parquet(out, index=False)
    meta = {
        "name": name,
        "kind": "oof",
        "labels": [str(x) for x in labels],
        "metric": METRIC,
        "saved": _now(),
        **(meta or {}),
    }
    (ART / "oof" / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def save_test(name: str, ids, probs: np.ndarray, labels: list) -> Path:
    probs = np.asarray(probs, dtype=np.float64)
    df = pd.DataFrame(probs, columns=prob_cols(labels))
    df.insert(0, ID, np.asarray(ids))
    out = ART / "test_preds" / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


def list_oof() -> list:
    return sorted(p.stem for p in (ART / "oof").glob("*.parquet"))


def list_test() -> list:
    return sorted(p.stem for p in (ART / "test_preds").glob("*.parquet"))


def load_oof_df(name: str) -> pd.DataFrame:
    return pd.read_parquet(ART / "oof" / f"{name}.parquet")


def load_test_df(name: str) -> pd.DataFrame:
    return pd.read_parquet(ART / "test_preds" / f"{name}.parquet")


def read_meta(name: str) -> dict:
    p = ART / "oof" / f"{name}.meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _matrix(df: pd.DataFrame, ids_order, labels: list) -> np.ndarray:
    cols = prob_cols(labels)
    m = df.set_index(ID).reindex(ids_order)[cols]
    if m.isna().any().any():
        raise ValueError("prediction matrix has NaN after id-align — ids don't match. "
                         "All models must use the same ids / folds.")
    return m.to_numpy(dtype=np.float64)


def blend_frame():
    """Everything the blender needs, aligned to train id order:
       (ids, y_idx, labels, {name: oof_probs (N,C)})."""
    train = load_train()
    ids = train[ID].to_numpy()
    labels = class_labels(train)
    y_idx = encode_target(train[TARGET].to_numpy(), labels)
    oof = {n: _matrix(load_oof_df(n), ids, labels) for n in list_oof()}
    return ids, y_idx, labels, oof


def test_frame(labels: list):
    """(test_ids, {name: test_probs (M,C)}) aligned to test id order, for the frozen blend."""
    test = load_test()
    ids = test[ID].to_numpy()
    preds = {n: _matrix(load_test_df(n), ids, labels) for n in list_test()}
    return ids, preds


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score(y_idx, probs, metric: str | None = None) -> float:
    return _metrics.score(y_idx, probs, metric or METRIC)


def greater_is_better(metric: str | None = None) -> bool:
    return _metrics.greater_is_better(metric or METRIC)


def cv_report(y_idx, oof_probs, fold_ids) -> dict:
    """Per-fold + overall OOF score. Returns {'overall', 'mean', 'std', 'per_fold'}."""
    y_idx = np.asarray(y_idx)
    per = []
    for f in sorted(np.unique(fold_ids)):
        m = fold_ids == f
        per.append(score(y_idx[m], oof_probs[m]))
    return {
        "overall": score(y_idx, oof_probs),
        "mean": float(np.mean(per)),
        "std": float(np.std(per)),
        "per_fold": [float(x) for x in per],
    }


# --------------------------------------------------------------------------------------
# Submission writing (matches sample_submission — prob columns OR a single label column)
# --------------------------------------------------------------------------------------
def write_submission(test_ids, probs: np.ndarray, labels: list, path: str | Path = None) -> Path:
    """Write submission.csv matching sample_submission's shape:
       * if it has one prob column per class -> write probabilities in that column order
       * if it has a single target column     -> write the argmax class label
    """
    ss = load_sample_submission()
    path = Path(path) if path else (ROOT / "submission.csv")
    out_cols = [c for c in ss.columns if c != ID]
    labels_str = [str(x) for x in labels]

    sub = pd.DataFrame({ID: np.asarray(test_ids)})
    if len(out_cols) == len(labels):
        # probability submission — map each sample_submission column to our canonical label order
        # try to match by class-name; fall back to positional order.
        if set(out_cols) == set(labels_str):
            for col in out_cols:
                sub[col] = probs[:, labels_str.index(col)]
        else:
            for j, col in enumerate(out_cols):
                sub[col] = probs[:, j]
    elif len(out_cols) == 1:
        # single predicted-label submission
        sub[out_cols[0]] = decode_target(probs.argmax(axis=1), labels)
    else:
        raise ValueError(f"can't map {probs.shape[1]} classes onto sample_submission cols {out_cols}")

    # order rows to match sample_submission's id order
    sub = ss[[ID]].merge(sub, on=ID, how="left")
    assert not sub.isna().any().any(), "submission has NaN / missing ids vs sample_submission"
    sub.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# Experiment scoreboard
# --------------------------------------------------------------------------------------
_LOG = ROOT / "EXPERIMENTS_LOG.md"
_LOG_HEADER = "| date | name | family | CV (mean±std) | LB | n_feat | notes |\n" \
              "|------|------|--------|---------------|----|--------|-------|\n"


def log_experiment(name: str, family: str, cv, cv_std=None, lb="", n_feat="", notes="") -> None:
    """Append one row to EXPERIMENTS_LOG.md. `cv` may be a float or a cv_report dict."""
    if isinstance(cv, dict):
        cv_std = cv.get("std", cv_std)
        cv = cv.get("mean", cv.get("overall"))
    cv_txt = f"{cv:.5f}" + (f"±{cv_std:.5f}" if cv_std is not None else "")
    row = f"| {_now()} | {name} | {family} | {cv_txt} | {lb} | {n_feat} | {notes} |\n"
    if not _LOG.exists() or "| date | name |" not in _LOG.read_text(encoding="utf-8", errors="ignore"):
        prefix = _LOG.read_text(encoding="utf-8") if _LOG.exists() else "# Experiments Log\n\n"
        _LOG.write_text(prefix + ("\n" if not prefix.endswith("\n") else "") + _LOG_HEADER + row,
                        encoding="utf-8")
    else:
        with _LOG.open("a", encoding="utf-8") as fh:
            fh.write(row)


if __name__ == "__main__":
    # quick self-check
    print("root:", ROOT)
    print("config:", {k: CFG[k] for k in ("competition", "id_col", "target_col", "metric",
                                          "n_folds", "fold_scheme", "seed")})
    try:
        print("train:", load_train().shape, "| test:", load_test().shape)
        print("classes:", class_labels())
    except FileNotFoundError as e:
        print("data not placed yet:", e)
