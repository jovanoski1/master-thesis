"""Thresholds and metrics (Sections 4.7 and 4.9).

Both thresholds are fixed on the validation block and then applied unchanged to the test
block: the F1-maximising threshold, and the score above which the highest-scoring
ALERT_BUDGET share of validation transactions lies.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def threshold_f1(y, s) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def threshold_budget(s, budget: float) -> float:
    """Score of the k-th highest transaction, k = ceil(budget * n)."""
    k = max(1, int(np.ceil(budget * len(s))))
    return float(np.partition(s, len(s) - k)[len(s) - k])


def at_threshold(y, s, tau) -> dict:
    flag = s >= tau
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "alert_rate": float(flag.mean())}


def top_k(y, s, frac: float, seed: int = 0) -> dict:
    """The highest-scoring ceil(frac * n) transactions of this block, ties broken at random
    with a fixed seed: a rank-based operating point unaffected by tied scores."""
    n = len(s)
    k = max(1, int(np.ceil(frac * n)))
    order = np.lexsort((np.random.default_rng(seed).random(n), -np.asarray(s)))
    tp = int(np.asarray(y)[order[:k]].sum())
    pos = int(np.asarray(y).sum())
    return {"k": k, "tp": tp, "recall": tp / pos if pos else 0.0, "precision": tp / k}


def evaluate(y, s, tau_f1, tau_budget, default=None, budget=None) -> dict:
    y = np.asarray(y)
    both = 0 < y.sum() < len(y)
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "ap": float(average_precision_score(y, s)) if both else None,
           "roc_auc": float(roc_auc_score(y, s)) if both else None,
           "at_f1": at_threshold(y, s, tau_f1),
           "at_budget": at_threshold(y, s, tau_budget)}
    if default is not None:
        out["at_default"] = at_threshold(y, s, default)
    if budget is not None and len(y):
        out["top_k"] = top_k(y, s, budget)
    return out
