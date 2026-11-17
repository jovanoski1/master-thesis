"""Hyperparameter search, final fits and evaluation, resumable across sessions.

For each variant, family and feature set: candidates are drawn once from the declared grid
(fixed seed), each is fitted on the search subsample of the training block and scored by
average precision on the validation block (the temporal equivalent of RandomizedSearchCV
with a PredefinedSplit); the best candidate is fitted on the full training block with
N_SEEDS seeds; thresholds are fixed on validation and the test block is scored once per
seed. Every candidate and every seed is appended to a JSON-lines file on Drive as soon as
it completes, so an interrupted session resumes where it stopped.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from . import config, models
from . import evaluation as ev
from . import model_params as MP

FEATURES_DIR = config.ROOT / "data" / "features"
STORE_DIR = config.RESULTS / "models"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Store:
    """Append-only JSON-lines records: search candidates, final fits, supplement."""

    def __init__(self, directory=STORE_DIR):
        self.dir = Path(directory)
        (self.dir / "scores").mkdir(parents=True, exist_ok=True)
        self.records = {n: self._load(n) for n in ("search", "final", "supplement")}

    def _load(self, name):
        recs, path = {}, self.dir / f"{name}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    recs[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue          # a line cut off by a crash: that run is repeated
        return recs

    def has(self, name, key):
        return key in self.records[name]

    def add(self, name, rec):
        with open(self.dir / f"{name}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=_json) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.records[name][rec["key"]] = rec


@dataclass
class VariantData:
    variant: str
    names: list
    sets: dict
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    X_tail: np.ndarray
    y_tail: np.ndarray
    search_idx: np.ndarray = field(repr=False, default=None)


def load_variant(variant, features_dir=FEATURES_DIR, results_dir=config.RESULTS,
                 search_fraction=MP.SEARCH_FRACTION, seed=MP.SEARCH_SEED) -> VariantData:
    """Training rows exclude the burn-in day; test rows exclude the tail, which is kept
    apart for a separate report. Set C is loaded once; A and B are column subsets."""
    definition = json.loads((Path(results_dir) / f"feature_sets_{variant}.json").read_text())
    names = definition["sets"]["C"]
    df = pd.read_parquet(Path(features_dir) / f"{variant}.parquet",
                         columns=["block", "is_laundering", "burn_in", "tail"] + names)
    block, y = df["block"].astype(str).to_numpy(), df["is_laundering"].to_numpy(np.int8)
    burn, tail = df["burn_in"].to_numpy(), df["tail"].to_numpy()
    M = df[names].to_numpy(np.float32)
    del df
    masks = {"tr": (block == "train") & ~burn, "va": block == "val",
             "te": (block == "test") & ~tail, "tail": tail}
    parts = {k: (M[m], y[m]) for k, m in masks.items()}
    del M
    n_tr = len(parts["tr"][1])
    rng = np.random.default_rng(seed)
    search_idx = np.sort(rng.choice(n_tr, size=int(round(search_fraction * n_tr)), replace=False))
    sets = {s: [names.index(c) for c in definition["sets"][s]] for s in MP.SETS}
    return VariantData(variant, names, sets, *parts["tr"], *parts["va"], *parts["te"],
                       *parts["tail"], search_idx=search_idx)


def candidates(family, n):
    return list(ParameterSampler(MP.GRIDS[family], n_iter=n, random_state=MP.SEARCH_SEED))


def _key(*parts):
    return "|".join(str(p) for p in parts)


def run_config(store: Store, data: VariantData, family: str, set_name: str,
               n_candidates: int, n_seeds: int, progress=print) -> dict:
    v, idx = data.variant, data.sets[set_name]
    names = [data.names[i] for i in idx]
    cands = candidates(family, n_candidates)
    fixed = MP.FIXED[family]

    # --- search on the training subsample, scored on validation -------------------------
    todo = [i for i in range(len(cands)) if not store.has("search", _key(v, family, set_name, f"cand{i}"))]
    if todo:
        Xs, ys = data.X_tr[data.search_idx][:, idx], data.y_tr[data.search_idx]
        Xv = data.X_va[:, idx]
        if family in models.SCALED:
            pre = models.Preprocessor(names).fit(Xs)
            Xs, Xv = pre.transform(Xs), pre.transform(Xv)
        for i in todo:
            tic = time.perf_counter()
            m = models.make(family, {**fixed, **cands[i]}, seed=0)
            models.fit(m, Xs, ys, Xv, data.y_va)
            ap = float(average_precision_score(data.y_va, models.scores(m, Xv)))
            secs = round(time.perf_counter() - tic, 1)
            store.add("search", {"key": _key(v, family, set_name, f"cand{i}"), "variant": v,
                                 "family": family, "set": set_name, "candidate": i,
                                 "params": cands[i], "val_ap": ap, "seconds": secs})
            progress(f"    {v} {family} set {set_name} candidate {i + 1}/{len(cands)}: "
                     f"val AP {ap:.4f} ({secs} s)")
        del Xs, Xv
    found = [store.records["search"][_key(v, family, set_name, f"cand{i}")] for i in range(len(cands))]
    best = max(found, key=lambda r: r["val_ap"])

    # --- final fits on the full training block -------------------------------------------
    seeds = [s for s in range(n_seeds) if not store.has("final", _key(v, family, set_name, f"seed{s}"))]
    if seeds:
        Xt, Xv, Xte, Xtl = (A[:, idx] for A in (data.X_tr, data.X_va, data.X_te, data.X_tail))
        if family in models.SCALED:
            pre = models.Preprocessor(names).fit(Xt)
            Xt, Xv, Xte, Xtl = (pre.transform(A) for A in (Xt, Xv, Xte, Xtl))
        for s in seeds:
            tic = time.perf_counter()
            m = models.make(family, {**fixed, **best["params"]}, seed=s)
            models.fit(m, Xt, data.y_tr, Xv, data.y_va)
            sv, ste = models.scores(m, Xv), models.scores(m, Xte)
            stl = models.scores(m, Xtl) if len(Xtl) else np.empty(0)
            tau_f1, tau_b = ev.threshold_f1(data.y_va, sv), ev.threshold_budget(sv, MP.ALERT_BUDGET)
            d = models.DEFAULT_THRESHOLD[family]
            rec = {"key": _key(v, family, set_name, f"seed{s}"), "variant": v, "family": family,
                   "set": set_name, "seed": s, "params": best["params"],
                   "search_val_ap": best["val_ap"], "n_features": len(idx),
                   "tau_f1": tau_f1, "tau_budget": tau_b,
                   "val": ev.evaluate(data.y_va, sv, tau_f1, tau_b, d, MP.ALERT_BUDGET),
                   "test": ev.evaluate(data.y_te, ste, tau_f1, tau_b, d, MP.ALERT_BUDGET),
                   "tail": ev.evaluate(data.y_tail, stl, tau_f1, tau_b, d) if len(stl) else None,
                   "seconds": round(time.perf_counter() - tic, 1)}
            if s == 0:
                rec["extras"] = models.extras(m, names)
            out = {"test": ste.astype(np.float32), "tail": stl.astype(np.float32)}
            if s == 0:
                out["val"] = sv.astype(np.float32)
            np.savez_compressed(store.dir / "scores" / f"{v}_{family}_{set_name}_seed{s}.npz", **out)
            store.add("final", rec)
            progress(f"    {v} {family} set {set_name} seed {s}: test AP {rec['test']['ap']:.4f}, "
                     f"recall at budget {rec['test']['at_budget']['recall']:.3f} ({rec['seconds']} s)")
    return best


def run_family(store, data_by_variant: dict, family: str, sets=MP.SETS,
               n_candidates=None, n_seeds=MP.N_SEEDS, progress=print):
    n_candidates = n_candidates or MP.N_CANDIDATES[family]
    for v, data in data_by_variant.items():
        for s in sets:
            progress(f"  {v}, {family}, set {s}")
            run_config(store, data, family, s, n_candidates, n_seeds, progress)
    return summary(store, family=family)


def summary(store, family=None) -> pd.DataFrame:
    rows = []
    for r in store.records["final"].values():
        if family and r["family"] != family:
            continue
        t = r["test"]
        rows.append({"variant": r["variant"], "family": r["family"], "set": r["set"], "seed": r["seed"],
                     "val AP (search)": r["search_val_ap"], "test AP": t["ap"],
                     "recall @ budget": t["at_budget"]["recall"],
                     "precision @ budget": t["at_budget"]["precision"],
                     "alert rate @ budget": t["at_budget"]["alert_rate"],
                     "recall @ top 0.1%": t["top_k"]["recall"],
                     "precision @ F1": t["at_f1"]["precision"], "recall @ F1": t["at_f1"]["recall"],
                     "F1": t["at_f1"]["f1"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keys = ["variant", "family", "set"]
    metrics = [c for c in df.columns if c not in keys + ["seed"]]
    agg = df.groupby(keys)[metrics].agg(["mean", "std"])
    agg[("seeds", "n")] = df.groupby(keys)["seed"].count()
    return agg


def timing_run(data: VariantData, families, set_name="C", progress=print) -> pd.DataFrame:
    """One fit per family on the search subsample (first declared candidate). Measures
    time only; no score is computed or shown."""
    rows = []
    idx = data.sets[set_name]
    names = [data.names[i] for i in idx]
    Xs, ys = data.X_tr[data.search_idx][:, idx], data.y_tr[data.search_idx]
    Xv = data.X_va[:, idx]
    Xs_scaled = Xv_scaled = None
    for f in families:
        if f in models.SCALED and Xs_scaled is None:
            pre = models.Preprocessor(names).fit(Xs)
            Xs_scaled, Xv_scaled = pre.transform(Xs), pre.transform(Xv)
        X1, X2 = (Xs_scaled, Xv_scaled) if f in models.SCALED else (Xs, Xv)
        tic = time.perf_counter()
        m = models.make(f, {**MP.FIXED[f], **candidates(f, 1)[0]}, seed=0)
        models.fit(m, X1, ys, X2, data.y_va)
        secs = time.perf_counter() - tic
        rows.append({"family": f, "seconds_per_search_fit": round(secs, 1)})
        progress(f"  {f}: {secs:.1f} s")
    return pd.DataFrame(rows).set_index("family")


def projection(timing: pd.DataFrame, n_train: dict, n_features: dict, ref_variant: str,
               n_candidates: dict, n_seeds: int, fraction: float) -> pd.DataFrame:
    """Projected hours, assuming time proportional to rows and to features."""
    rows = []
    for f, t in timing["seconds_per_search_fit"].items():
        total = 0.0
        for v in n_train:
            for s, k in n_features[v].items():
                scale = (n_train[v] / n_train[ref_variant]) * (k / n_features[ref_variant]["C"])
                total += t * scale * (n_candidates[f] + n_seeds / fraction)
        rows.append({"family": f, "candidates": n_candidates[f], "projected_hours": total / 3600})
    out = pd.DataFrame(rows).set_index("family")
    out.loc["total"] = [out["candidates"].sum(), out["projected_hours"].sum()]
    return out.round(2)


def best_configuration(store, variant) -> dict:
    """Highest mean validation AP over complete final fits (validation, never test)."""
    df = pd.DataFrame([{"family": r["family"], "set": r["set"], "val_ap": r["val"]["ap"]}
                       for r in store.records["final"].values() if r["variant"] == variant])
    g = df.groupby(["family", "set"])["val_ap"].agg(["mean", "count"])
    g = g[g["count"] == g["count"].max()]
    family, set_name = g["mean"].idxmax()
    return {"family": family, "set": set_name, "mean_val_ap": float(g["mean"].max())}


def run_supplement(store, data: VariantData, n_seeds=MP.N_SEEDS, progress=print) -> pd.DataFrame:
    """Section 5.7: the best configuration without class weighting, with weighting alone
    (default threshold), and with weighting and the tuned threshold (the main result)."""
    best = best_configuration(store, data.variant)
    family, set_name = best["family"], best["set"]
    params = store.records["final"][_key(data.variant, family, set_name, "seed0")]["params"]
    idx = data.sets[set_name]
    names = [data.names[i] for i in idx]
    Xt, Xv, Xte = (A[:, idx] for A in (data.X_tr, data.X_va, data.X_te))
    if family in models.SCALED:
        pre = models.Preprocessor(names).fit(Xt)
        Xt, Xv, Xte = (pre.transform(A) for A in (Xt, Xv, Xte))
    d = models.DEFAULT_THRESHOLD[family]
    for cond, cw in (("unweighted, default threshold", None), ("weighted, default threshold", "balanced")):
        for s in range(n_seeds):
            key = _key(data.variant, family, set_name, cond, f"seed{s}")
            if store.has("supplement", key):
                continue
            m = models.make(family, {**MP.FIXED[family], **params}, seed=s, class_weight=cw)
            models.fit(m, Xt, data.y_tr, Xv, data.y_va)
            ste = models.scores(m, Xte)
            res = ev.evaluate(data.y_te, ste, d, d)
            store.add("supplement", {"key": key, "variant": data.variant, "family": family,
                                     "set": set_name, "condition": cond, "seed": s,
                                     "test_ap": res["ap"], **res["at_f1"]})
            progress(f"    {cond}, seed {s}: F1 {res['at_f1']['f1']:.3f}")
    rows = [r for r in store.records["supplement"].values() if r["variant"] == data.variant]
    for r in store.records["final"].values():
        if (r["variant"], r["family"], r["set"]) == (data.variant, family, set_name):
            rows.append({"condition": "weighted, tuned threshold (main result)", "seed": r["seed"],
                         "test_ap": r["test"]["ap"], **r["test"]["at_f1"]})
    df = pd.DataFrame(rows)
    cols = ["test_ap", "precision", "recall", "f1", "alert_rate"]
    out = df.groupby("condition")[cols].agg(["mean", "std"])
    out.attrs["configuration"] = best
    return out
