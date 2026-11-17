"""Notebook 02 pipeline: Sets A, B and C, causality verification, screening, persistence.

Usage:  python -m aml.build_features --variant HI-Small
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import aggregates as agg
from . import config, fx, graph, screening
from . import feature_params as P
from .features import base_attributes

FEATURES_DIR = config.ROOT / "data" / "features"
ONE_HOT = {"payment_currency": "cur_pay", "receiving_currency": "cur_rec",
           "payment_format": "fmt"}
READ_COLS = ["Timestamp", "src_account", "dst_account", "From Bank", "To Bank",
             "Amount Paid", "Amount Received", "Payment Currency", "Receiving Currency",
             "Payment Format", "Is Laundering", "block", "file_row"]
WRITE_CHUNK = 500_000        # rows per Parquet write, to bound memory


def tail_flag(ts: pd.Series, share: float) -> tuple[np.ndarray, pd.Timestamp]:
    """Flag rows after the last calendar day whose transaction count reaches `share` of the
    median daily count. Uses timestamps only, so no label informs the rule."""
    day = ts.dt.floor("D")
    counts = day.value_counts().sort_index()
    last_regular = counts.index[(counts >= share * counts.median()).to_numpy()].max()
    return (day > last_regular).to_numpy(), last_regular


def set_a(df: pd.DataFrame, rates: pd.Series, train: np.ndarray,
          day_of_week: str = "weekend") -> pd.DataFrame:
    """Set A, encoded: amounts in the base currency, binary flags, hour, the day-of-week
    variant chosen in feature_params, and one indicator per category observed in the
    training block (unseen categories in later blocks are all-zero, so no information
    from them reaches the encoding)."""
    raw = base_attributes(df)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    out = {"amount_paid_base": f32(fx.to_base(df["Amount Paid"], df["Payment Currency"], rates)),
           "amount_received_base": f32(fx.to_base(df["Amount Received"],
                                                  df["Receiving Currency"], rates)),
           "currency_mismatch": f32(raw["currency_mismatch"]),
           "cross_bank": f32(raw["cross_bank"]),
           "hour": f32(raw["hour"])}
    if day_of_week == "weekend":
        out["weekend"] = f32(raw["day_of_week"].to_numpy() >= 5)
    elif day_of_week == "full":
        out["day_of_week"] = f32(raw["day_of_week"])
    elif day_of_week != "drop":
        raise ValueError(f"DAY_OF_WEEK must be 'weekend', 'drop' or 'full', not {day_of_week!r}")
    for col, prefix in ONE_HOT.items():
        s = raw[col]
        codes = s.cat.codes.to_numpy()
        for c in s[train].cat.remove_unused_categories().cat.categories:
            out[f"{prefix}={str(c).strip().replace(' ', '_')}"] = \
                f32(codes == s.cat.categories.get_loc(c))
    del raw
    return pd.DataFrame(out)


def build(variant: str, interim_dir=config.DATA_INTERIM, features_dir=FEATURES_DIR,
          results_dir=config.RESULTS, progress=print) -> dict:
    t_start = time.perf_counter()
    interim_dir, features_dir, results_dir = map(Path, (interim_dir, features_dir, results_dir))
    tables = results_dir / "tables"
    for d in (features_dir, tables):
        d.mkdir(parents=True, exist_ok=True)
    timing = {}

    df = pd.read_parquet(interim_dir / f"{variant}.parquet", columns=READ_COLS)
    origin = df["Timestamp"].min()
    t = agg.minutes(df["Timestamp"], origin)
    train = (df["block"] == "train").to_numpy()
    burn_in = t < agg.window_minutes(P.BURN_IN)
    tail, last_regular_day = tail_flag(df["Timestamp"], P.TAIL_SHARE)
    rng = np.random.default_rng(P.SEED)

    tic = time.perf_counter()
    rates, fx_diag, fx_pairs = fx.implied_rates(df.loc[train], P.BASE_CURRENCY)
    fx_diag.to_csv(tables / f"{variant}_fx_rates.csv")
    fx_pairs.to_csv(tables / f"{variant}_fx_pairs.csv", index=False)
    A = set_a(df, rates, train, P.DAY_OF_WEEK)
    timing["set_a"] = round(time.perf_counter() - tic, 1)
    progress(f"Set A: {A.shape[1]} columns ({timing['set_a']} s)")

    tic = time.perf_counter()
    src, dst, n_acc = agg.account_codes(df)
    n = len(df)
    file_row = df["file_row"].to_numpy()
    block = pd.Categorical(df["block"])
    label = df["Is Laundering"].to_numpy(np.int8)
    tail_info = {"last_regular_day": str(last_regular_day.date()), "n_rows": int(tail.sum()),
                 "n_laundering": int(label[tail].sum()),
                 "blocks": pd.Series(block[tail]).astype(str).value_counts().to_dict()}
    del df                                            # only arrays are needed from here on
    gc.collect()
    w = agg.window_minutes(P.WINDOW)
    amt_out = np.nan_to_num(A["amount_paid_base"].to_numpy(np.float64))
    amt_in = np.nan_to_num(A["amount_received_base"].to_numpy(np.float64))
    B = agg.set_b(src, dst, t, amt_out, amt_in, w, n_acc)
    timing["set_b"] = round(time.perf_counter() - tic, 1)
    progress(f"Set B: {B.shape[1]} columns ({timing['set_b']} s)")

    # One float32 matrix for all features; each part is released as soon as it is copied,
    # so the peak memory is the matrix plus one part rather than every copy at once.
    cols_by_set = {"A": list(A.columns), "B": list(B.columns),
                   "C": [f"{r}_{st}" for r in graph.ROLES for st in graph.STATS]}
    order = cols_by_set["A"] + cols_by_set["B"] + cols_by_set["C"]
    M = np.empty((n, len(order)), dtype=np.float32, order="F")
    j = 0
    for part in (A, B):
        for c in part.columns:
            M[:, j] = part[c].to_numpy()
            j += 1
    del A, B, part
    gc.collect()

    tic = time.perf_counter()
    sw = agg.window_minutes(P.SNAPSHOT_WINDOW)
    C, snaps, snap_of_row = graph.set_c(src, dst, t, origin, P.SNAPSHOT_FREQ, sw, progress)
    assert list(C.columns) == cols_by_set["C"]
    for c in C.columns:
        M[:, j] = C[c].to_numpy()
        j += 1
    del C
    gc.collect()
    snaps.to_csv(tables / f"{variant}_snapshots.csv", index=False)
    timing["set_c"] = round(time.perf_counter() - tic, 1)
    progress(f"Set C: {len(cols_by_set['C'])} columns ({timing['set_c']} s)")
    X = pd.DataFrame(M, columns=order, copy=False)          # a view, not a copy
    view = lambda k: X[cols_by_set[k]]

    tic = time.perf_counter()
    vrows = rng.choice(n, size=min(P.VERIFY_SAMPLE, n), replace=False)
    verification = {
        "set_b": agg.verify_b(src, dst, t, amt_out, amt_in, view("B"), w, vrows),
        "set_c": graph.verify_c(src, dst, t, view("C"), snap_of_row, sw,
                                vrows[: max(200, len(vrows) // 4)]),
        "snapshots_use_only_earlier_edges": bool(snaps["edges_strictly_before"].all()),
    }
    timing["verification"] = round(time.perf_counter() - tic, 1)
    if not (verification["set_b"]["passed"] and verification["set_c"]["passed"]
            and verification["snapshots_use_only_earlier_edges"]):
        raise RuntimeError(f"Causality verification failed: {verification}")
    progress(f"Causality verification passed ({timing['verification']} s)")

    fit_rows = np.flatnonzero(train & ~burn_in)
    sample = np.sort(rng.choice(fit_rows, size=min(P.SCREEN_SAMPLE, len(fit_rows)),
                                replace=False))
    kept, screen_report, rho = screening.screen(X.iloc[sample], order, P.SCREEN_THRESHOLD)
    rho.to_csv(tables / f"{variant}_feature_spearman.csv")
    sets = {"A": [c for c in cols_by_set["A"] if c in kept]}
    sets["B"] = sets["A"] + [c for c in cols_by_set["B"] if c in kept]
    sets["C"] = sets["B"] + [c for c in cols_by_set["C"] if c in kept]
    progress(f"Screening: {len(kept)} of {len(order)} columns kept; "
             f"sets A/B/C = {len(sets['A'])}/{len(sets['B'])}/{len(sets['C'])}")

    tic = time.perf_counter()
    writer = None
    for start in range(0, n, WRITE_CHUNK):
        sl = slice(start, min(start + WRITE_CHUNK, n))
        chunk = pd.DataFrame({"file_row": file_row[sl], "block": block[sl],
                              "is_laundering": label[sl], "burn_in": burn_in[sl],
                              "tail": tail[sl], "snapshot_minute": snap_of_row[sl],
                              **{c: M[sl, k] for k, c in enumerate(order)}})
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(features_dir / f"{variant}.parquet", table.schema)
        writer.write_table(table)
    writer.close()
    timing["write"] = round(time.perf_counter() - tic, 1)

    definition = {"variant": variant, "sets": sets,
                  "columns_before_screening": cols_by_set,
                  "parameters": {k: getattr(P, k) for k in dir(P) if k.isupper()}}
    (results_dir / f"feature_sets_{variant}.json").write_text(json.dumps(definition, indent=2))

    report = {
        "variant": variant,
        "n_rows": int(n), "n_burn_in_rows": int(burn_in.sum()),
        "n_training_rows_after_burn_in": int(len(fit_rows)),
        "tail": tail_info,
        "n_columns": {k: len(v) for k, v in cols_by_set.items()},
        "set_sizes_after_screening": {k: len(v) for k, v in sets.items()},
        "screening": screen_report,
        "fx": {"n_currencies": int(len(fx_diag)),
               "methods": fx_diag["method"].value_counts().to_dict(),
               "max_pair_iqr_log_rate": float(fx_pairs["iqr"].max()) if len(fx_pairs) else None},
        "snapshots": {"n": int(len(snaps)),
                      "rows_without_snapshot": int((snap_of_row < 0).sum())},
        "verification": verification,
        "timing_seconds": {**timing, "total": round(time.perf_counter() - t_start, 1)},
    }
    (results_dir / f"features_{variant}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=config.VARIANTS, action="append")
    for v in ap.parse_args().variant or list(config.VARIANTS):
        print(json.dumps(build(v), indent=2, default=str))


if __name__ == "__main__":
    main()
