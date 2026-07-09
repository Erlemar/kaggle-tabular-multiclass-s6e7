"""
metrics.py — multiclass metric dispatch.

Every score in the kit goes through here so the competition metric is defined in exactly one
place. `y_idx` is the integer-encoded true label (0..C-1) in the canonical class order;
`probs` is an (N, C) array of predicted probabilities in that same class order.

To add a metric: write a function `(y_idx, probs) -> float` and register it in METRICS with
`greater_is_better`.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score, cohen_kappa_score


def multiclass_logloss(y_idx: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    p = np.clip(probs, 1e-15, 1 - 1e-15)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(y_idx, p, labels=list(range(n_classes))))


def accuracy(y_idx: np.ndarray, probs: np.ndarray) -> float:
    return float(accuracy_score(y_idx, probs.argmax(axis=1)))

def balanced_accuracy(y_idx: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_idx, probs.argmax(axis=1)))


def macro_f1(y_idx: np.ndarray, probs: np.ndarray) -> float:
    return float(f1_score(y_idx, probs.argmax(axis=1), average="macro"))


def qwk(y_idx: np.ndarray, probs: np.ndarray) -> float:
    """Quadratic weighted kappa on the argmax (ordinal targets). For a real QWK push, regress
    and optimize cut points at blend time — this is the classification proxy."""
    return float(cohen_kappa_score(y_idx, probs.argmax(axis=1), weights="quadratic"))


# name -> (fn, greater_is_better)
METRICS = {
    "logloss":  (multiclass_logloss, False),
    "accuracy": (accuracy,           True),
    "balanced_accuracy": (balanced_accuracy, True),
    "macro_f1": (macro_f1,           True),
    "qwk":      (qwk,                True),
}


def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"unknown metric '{name}'; known: {list(METRICS)}. Add it to src/metrics.py.")
    return METRICS[name]


def score(y_idx: np.ndarray, probs: np.ndarray, name: str) -> float:
    fn, _ = get_metric(name)
    return fn(np.asarray(y_idx), np.asarray(probs, dtype=np.float64))


def greater_is_better(name: str) -> bool:
    return get_metric(name)[1]
