"""Loading, typing and integrity checks for one transaction file (Section 4.4)."""
import csv

import numpy as np
import pandas as pd

from . import config

AMOUNT_COLS = ["Amount Received", "Amount Paid"]
# Arrow-backed strings use a fraction of the memory of Python objects; RAM is the
# binding constraint at five to seven million records (Section 4.4).
STR_DTYPE = "string[pyarrow]"
CATEGORY_COLS = ["src_account", "dst_account", "From Bank", "To Bank",
                 "Payment Format", "Receiving Currency", "Payment Currency"]


def verify_header(path) -> None:
    """Refuse to rename by position unless the file has the documented column order.

    Positional renaming alone does not protect against a reordered file; checking the
    raw header first is what makes the renaming in Listing 4.1 safe.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    if header != config.RAW_HEADER:
        raise ValueError(f"Unexpected header in {path}:\n{header}\n"
                         f"expected:\n{config.RAW_HEADER}")


def normalise_bank(s: pd.Series) -> pd.Series:
    """Bank identifiers may be zero-padded differently across files ("0110" vs "110")."""
    s = s.astype(STR_DTYPE).str.strip().str.lstrip("0")
    return s.where(s != "", "0")


def read_transactions(path) -> pd.DataFrame:
    """Read identifiers as strings and amounts as float64.

    Categorical typing and float32 downcasting are deferred to `finalise_types`, so
    that the join with the patterns file (exact equality on every field) happens
    before any lossy conversion.
    """
    verify_header(path)
    dtypes = {c: STR_DTYPE for c in config.COLUMNS
              if c not in AMOUNT_COLS + ["Is Laundering"]}
    dtypes.update({c: "float64" for c in AMOUNT_COLS})
    dtypes["Is Laundering"] = "int32"  # int8 sums can overflow silently in groupby
    df = pd.read_csv(path, header=0, names=config.COLUMNS, dtype=dtypes)
    # errors="coerce" is not silent: unparsable timestamps are counted in the
    # integrity report rather than dropped.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=config.TIMESTAMP_FORMAT,
                                     errors="coerce")
    df["From Bank"] = normalise_bank(df["From Bank"])
    df["To Bank"] = normalise_bank(df["To Bank"])
    df["file_row"] = np.arange(len(df), dtype=np.int32)  # audit trail to the raw file
    return df


def finalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Memory-oriented typing (Listing 4.1) and a stable chronological sort.

    The sort is stable, so transactions sharing a timestamp keep their file order
    and every rebuild yields the same row order.
    """
    for c in CATEGORY_COLS:
        df[c] = df[c].astype("category")
    for c in AMOUNT_COLS:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)


def _rel_dev(observed: float, published: float) -> float:
    return float(observed / published - 1.0)


def integrity_checks(df: pd.DataFrame, variant: str) -> dict:
    """The three checks of Section 4.4. Nothing is removed; every count is reported."""
    pub = config.PUBLISHED[variant]
    n = int(len(df))
    n_pos = int(df["Is Laundering"].sum())
    ratio = n / n_pos if n_pos else float("inf")
    ts = df["Timestamp"]

    report = {
        "n_records": n,
        "n_laundering": n_pos,
        "observed_ratio_1_in": round(ratio, 1),
        "published": pub,
        "dev_records": round(_rel_dev(n, pub["transactions"]), 4),
        "dev_laundering": round(_rel_dev(n_pos, pub["laundering"]), 4),
        "dev_ratio": round(_rel_dev(ratio, pub["ratio_1_in"]), 4),
        "n_unparsed_timestamps": int(ts.isna().sum()),
        "timestamp_non_decreasing": bool(ts.dropna().is_monotonic_increasing),
        "n_duplicate_records": int(df.duplicated(subset=config.COLUMNS).sum()),
        "n_nonpositive_amount_paid": int((df["Amount Paid"] <= 0).sum()),
        "first_timestamp": str(ts.min()),
        "last_timestamp": str(ts.max()),
        "span_days": round((ts.max() - ts.min()) / pd.Timedelta(days=1), 3),
    }
    report["checks"] = {
        "records_match_table_4_1": abs(report["dev_records"]) <= config.COUNT_TOLERANCE,
        "laundering_matches_table_4_1": abs(report["dev_laundering"]) <= config.COUNT_TOLERANCE,
        "ratio_matches_table_4_1": abs(report["dev_ratio"]) <= config.RATIO_TOLERANCE,
        "timestamps_parsed_and_ordered": (report["n_unparsed_timestamps"] == 0
                                          and report["timestamp_non_decreasing"]),
        "no_duplicate_records": report["n_duplicate_records"] == 0,
    }
    return report
