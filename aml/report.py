"""Notebook 04: tables, rank-based comparisons, paired bootstrap, typologies (Section 4.9).

Nothing here fits a model. Everything is computed from the records and scores that
Notebook 03 stored, together with the labels and typologies of the test rows.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from . import config

FAMILY_NAMES = {"logreg": "Logistic regression", "svm": "Linear SVM", "rf": "Random forest",
                "hgb": "Gradient boosting", "ffnn": "Neural network"}
FAMILIES = list(FAMILY_NAMES)
SETS = ["A", "B", "C"]
# Critical values q_0.05 of the Nemenyi test (Demsar, 2006, Table 5), by number of treatments.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


# --- records -------------------------------------------------------------------------------
def load_final(store_dir) -> pd.DataFrame:
    rows = []
    for line in (Path(store_dir) / "final.jsonl").read_text().splitlines():
        r = json.loads(line)
        t, v = r["test"], r["val"]
        rows.append({"variant": r["variant"], "family": r["family"], "set": r["set"],
                     "seed": r["seed"], "val_ap": v["ap"], "test_ap": t["ap"],
                     "test_roc_auc": t["roc_auc"], "recall_top": t["top_k"]["recall"],
                     "precision_top": t["top_k"]["precision"],
                     "recall_budget": t["at_budget"]["recall"],
                     "alert_rate_budget": t["at_budget"]["alert_rate"],
                     "precision_f1": t["at_f1"]["precision"], "recall_f1": t["at_f1"]["recall"],
                     "f1": t["at_f1"]["f1"], "alert_rate_f1": t["at_f1"]["alert_rate"],
                     "tau_f1": r["tau_f1"], "seconds": r["seconds"],
                     "n_pos": t["n_pos"], "n": t["n"],
                     "tail_tp": r["tail"]["at_f1"]["tp"] if r.get("tail") else 0,
                     "tail_fp": r["tail"]["at_f1"]["fp"] if r.get("tail") else 0,
                     "tail_fn": r["tail"]["at_f1"]["fn"] if r.get("tail") else 0,
                     "test_tp": t["at_f1"]["tp"], "test_fp": t["at_f1"]["fp"],
                     "test_fn": t["at_f1"]["fn"], "params": r["params"],
                     "extras": r.get("extras")})
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame, metric: str, digits: int = 3) -> pd.DataFrame:
    """'mean (sd)' strings, variant x family rows, set columns."""
    g = df.groupby(["variant", "family", "set"])[metric].agg(["mean", "std"])
    s = g["mean"].map(f"{{:.{digits}f}}".format) + " (" + g["std"].fillna(0).map(f"{{:.{digits}f}}".format) + ")"
    out = s.unstack("set")
    order = [(v, f) for v in out.index.get_level_values(0).unique() for f in FAMILIES
             if (v, f) in out.index]
    return out.loc[order]


# --- rank-based comparison (Demsar, 2006) ---------------------------------------------------
def friedman_nemenyi(table: pd.DataFrame) -> dict:
    """Rows are blocks (data conditions), columns are treatments; higher is better."""
    ranks = table.rank(axis=1, ascending=False)
    k, n = table.shape[1], table.shape[0]
    stat, p = stats.friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"statistic": float(stat), "p_value": float(p), "n_blocks": n, "k": k,
            "average_ranks": ranks.mean().sort_values().to_dict(), "critical_difference": float(cd),
            "ranks": ranks}


# --- paired bootstrap of average precision ---------------------------------------------------
class SortedScores:
    """A score vector sorted once, so that average precision under any row weights is
    computed in one pass (the bootstrap re-weights rows instead of copying them). Only the
    thresholds at which a positive appears change recall, so precision is evaluated there."""

    def __init__(self, y, s):
        s = np.asarray(s, dtype=np.float64)
        self.order = np.argsort(-s, kind="mergesort")
        ss = s[self.order]
        ends = np.append(np.flatnonzero(np.diff(ss) != 0), len(ss) - 1)  # last row of each tie group
        self.pos_rows = np.flatnonzero(np.asarray(y, dtype=bool)[self.order])
        group_end = ends[np.searchsorted(ends, self.pos_rows)]
        self.last = np.append(np.flatnonzero(np.diff(group_end) != 0), len(group_end) - 1)
        self.group_end = group_end[self.last]

    def ap(self, w=None) -> float:
        """Average precision as in scikit-learn (step-wise, ties as one threshold)."""
        ws = np.ones(len(self.order)) if w is None else w[self.order]
        cum_all = np.cumsum(ws)[self.group_end]
        cum_tp = np.cumsum(ws[self.pos_rows])[self.last]
        if len(cum_tp) == 0 or cum_tp[-1] <= 0:
            return np.nan
        prec = np.divide(cum_tp, cum_all, out=np.zeros_like(cum_tp), where=cum_all > 0)
        d_rec = np.diff(np.concatenate([[0.0], cum_tp])) / cum_tp[-1]
        return float(np.sum(d_rec * prec))


def paired_bootstrap(y, scores: dict, contrasts: list, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """scores[(group, set, seed)] -> test scores. For each contrast (group, high, low), the
    statistic is the seed-averaged difference AP(high) - AP(low). Every replicate draws one
    Poisson(1) weight per test row and applies it to all score vectors, so the comparison
    is paired across sets and seeds. Returns the observed difference and a percentile
    95 % interval of the bootstrap distribution."""
    y = np.asarray(y)
    prepared = {k: SortedScores(y, s) for k, s in scores.items()}
    seeds_of = {}
    for (g, st, sd) in scores:
        seeds_of.setdefault((g, st), []).append(sd)

    def stat(w):
        ap = {k: p.ap(w) for k, p in prepared.items()}
        out = {}
        for g, hi, lo in contrasts:
            common = sorted(set(seeds_of[(g, hi)]) & set(seeds_of[(g, lo)]))
            out[(g, hi, lo)] = np.mean([ap[(g, hi, s)] - ap[(g, lo, s)] for s in common])
        return out

    observed = stat(None)
    rng = np.random.default_rng(seed)
    draws = {c: np.empty(n_boot) for c in contrasts}
    for b in range(n_boot):
        w = rng.poisson(1.0, size=len(y)).astype(np.float64)
        for c, v in stat(w).items():
            draws[c][b] = v
    rows = []
    for (g, hi, lo) in contrasts:
        d = draws[(g, hi, lo)]
        lo_ci, hi_ci = np.nanpercentile(d, [2.5, 97.5])
        rows.append({"group": g, "contrast": f"{hi} - {lo}", "difference": observed[(g, hi, lo)],
                     "ci_low": lo_ci, "ci_high": hi_ci,
                     "share_of_replicates_above_0": float(np.nanmean(d > 0)),
                     "interval_excludes_0": bool(lo_ci > 0 or hi_ci < 0)})
    return pd.DataFrame(rows)


# --- labels, typologies and scores of the test rows ----------------------------------------
def test_rows(variant, features_dir=config.ROOT / "data" / "features",
              interim_dir=config.DATA_INTERIM) -> dict:
    """Labels and typologies in the row order used by Notebook 03 (test block without the
    tail, and the tail), with the alignment of the two files checked on file_row."""
    f = pd.read_parquet(Path(features_dir) / f"{variant}.parquet",
                        columns=["file_row", "block", "is_laundering", "tail"])
    i = pd.read_parquet(Path(interim_dir) / f"{variant}.parquet", columns=["file_row", "typology"])
    if not np.array_equal(f["file_row"].to_numpy(), i["file_row"].to_numpy()):
        raise RuntimeError("feature and interim rows are not aligned")
    block, tail = f["block"].astype(str).to_numpy(), f["tail"].to_numpy()
    test, tl = (block == "test") & ~tail, tail
    typ = i["typology"].cat.add_categories("no pattern").fillna("no pattern").astype(str).to_numpy()
    y = f["is_laundering"].to_numpy(np.int8)
    return {"y_test": y[test], "typology_test": typ[test], "y_tail": y[tl]}


def load_scores(store_dir, variant, family, set_name, seed):
    return np.load(Path(store_dir) / "scores" / f"{variant}_{family}_{set_name}_seed{seed}.npz")


def typology_recall(y, typology, scores_by_seed: dict, taus: dict, budget: float) -> pd.DataFrame:
    """Per typology: laundering transactions in the test block, share detected at the F1
    threshold and share among the highest-scoring `budget` share (both averaged over seeds)."""
    lab = y == 1
    k = max(1, int(np.ceil(budget * len(y))))
    det_f1, det_top = [], []
    for sd, s in scores_by_seed.items():
        det_f1.append(s >= taus[sd])
        top = np.zeros(len(s), dtype=bool)
        order = np.lexsort((np.random.default_rng(0).random(len(s)), -s))
        top[order[:k]] = True
        det_top.append(top)
    det_f1, det_top = np.mean(det_f1, axis=0), np.mean(det_top, axis=0)
    t = pd.DataFrame({"typology": typology[lab], "f1": det_f1[lab], "top": det_top[lab]})
    out = t.groupby("typology").agg(n=("f1", "size"), recall_at_f1=("f1", "mean"),
                                    recall_in_top=("top", "mean"))
    return out.sort_values("n", ascending=False)


def with_tail(df: pd.DataFrame) -> pd.DataFrame:
    """F1 at the validation F1 threshold, on the test block alone and with the tail added."""
    d = df.copy()
    tp, fp, fn = d.test_tp + d.tail_tp, d.test_fp + d.tail_fp, d.test_fn + d.tail_fn
    p, r = tp / (tp + fp).clip(lower=1), tp / (tp + fn).clip(lower=1)
    d["f1_with_tail"] = np.where(p + r > 0, 2 * p * r / (p + r).where(p + r > 0, 1), 0.0)
    return d.groupby(["variant", "family", "set"])[["f1", "f1_with_tail"]].mean()


def ap_with_tail(store_dir, variant, family, set_name, seeds, y_test, y_tail) -> float:
    vals = []
    for sd in seeds:
        sc = load_scores(store_dir, variant, family, set_name, sd)
        vals.append(average_precision_score(np.concatenate([y_test, y_tail]),
                                            np.concatenate([sc["test"], sc["tail"]])))
    return float(np.mean(vals))
