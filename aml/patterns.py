"""Parser for the laundering-patterns file and its join to the transactions (Section 4.4).

The typology label produced here is descriptive only (Section 5.5). It must never
enter a feature matrix; see config.NON_FEATURE_COLS.
"""
import csv

import numpy as np
import pandas as pd

from . import config
from .load import AMOUNT_COLS, STR_DTYPE, normalise_bank

TYPOLOGIES = {
    "FAN-OUT": "fan-out", "FAN-IN": "fan-in",
    "GATHER-SCATTER": "gather-scatter", "SCATTER-GATHER": "scatter-gather",
    "CYCLE": "cycle", "RANDOM": "random", "BIPARTITE": "bipartite", "STACK": "stack",
}
JOIN_COLS = config.COLUMNS[:-1]   # every recorded field except the label


def parse_patterns(path) -> pd.DataFrame:
    """Read 'BEGIN LAUNDERING ATTEMPT - <TYPE>[: detail]' ... 'END ...' blocks."""
    records, attempt, typology = [], -1, None
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                label = line.split("-", 1)[1].split(":", 1)[0].strip().upper()
                typology = TYPOLOGIES.get(label, label.lower())  # unknown labels are reported
            elif line.startswith("END LAUNDERING ATTEMPT"):
                typology = None
            else:
                if typology is None:
                    raise ValueError(f"{path}:{lineno}: transaction outside an attempt block")
                fields = next(csv.reader([line]))
                if len(fields) != len(config.COLUMNS):
                    raise ValueError(f"{path}:{lineno}: expected {len(config.COLUMNS)} "
                                     f"fields, found {len(fields)}")
                records.append(fields + [attempt, typology])

    pat = pd.DataFrame(records, columns=config.COLUMNS + ["attempt_id", "typology"])
    pat["Timestamp"] = pd.to_datetime(pat["Timestamp"], format=config.TIMESTAMP_FORMAT)
    for c in AMOUNT_COLS:
        pat[c] = pat[c].astype("float64")
    pat["Is Laundering"] = pat["Is Laundering"].astype("int32")
    pat["From Bank"] = normalise_bank(pat["From Bank"])
    pat["To Bank"] = normalise_bank(pat["To Bank"])
    for c in JOIN_COLS:   # identical dtypes on both sides of the join
        if c not in AMOUNT_COLS + ["Timestamp"]:
            pat[c] = pat[c].astype(STR_DTYPE)
    return pat


def attach_typology(df: pd.DataFrame, pat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach typology and attempt id to the transactions by exact equality of every field.

    Only rows sharing both a timestamp and an originating account with some pattern
    transaction can match, so the exact join runs on that small candidate subset
    rather than on all records. The result is identical to a full left join.
    """
    n = len(df)
    pmap = pat[JOIN_COLS + ["attempt_id", "typology"]]
    n_multi = int(pmap.duplicated(JOIN_COLS).sum())   # listed in more than one attempt
    pmap = pmap.drop_duplicates(JOIN_COLS, keep="first")

    cand = (df["Timestamp"].isin(pmap["Timestamp"].unique())
            & df["src_account"].isin(pmap["src_account"].unique())).to_numpy()
    sub = df.loc[cand, JOIN_COLS].copy()
    sub["_pos"] = np.flatnonzero(cand)
    m = sub.merge(pmap, on=JOIN_COLS, how="inner", validate="many_to_one")
    pos = m["_pos"].to_numpy()

    attempt = pd.Series(pd.NA, index=df.index, dtype="Int32")
    attempt.iloc[pos] = m["attempt_id"].to_numpy(dtype="int32")
    typology = np.full(n, None, dtype=object)
    typology[pos] = m["typology"].to_numpy(dtype=object)
    df["attempt_id"] = attempt
    df["typology"] = pd.Categorical(typology)

    label = df["Is Laundering"].to_numpy()
    has_typ = df["typology"].notna().to_numpy()
    report = {
        "n_pattern_rows": int(len(pat)),
        "n_attempts": int(pat["attempt_id"].nunique()),
        "attempts_by_typology": (pat.drop_duplicates("attempt_id")["typology"]
                                 .value_counts().to_dict()),
        "unknown_typology_labels": sorted(set(pat["typology"]) - set(TYPOLOGIES.values())),
        "n_pattern_transactions_in_multiple_attempts": n_multi,
        "n_pattern_transactions_unmatched": int(len(pmap)
                                                - len(m[JOIN_COLS].drop_duplicates())),
        "n_rows_with_typology": int(has_typ.sum()),
        "n_laundering_rows_without_typology": int(((label == 1) & ~has_typ).sum()),
        "n_typology_rows_labelled_licit": int((has_typ & (label == 0)).sum()),
    }
    return df, report
