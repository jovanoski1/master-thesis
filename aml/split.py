"""Chronological partitioning, fixed once and never revised (Section 4.5, Listing 4.2)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

BLOCKS = ["train", "val", "test"]


def compute_boundaries(ts: pd.Series, q=config.SPLIT_QUANTILES) -> dict:
    """Quantiles of the record timestamps, so the blocks hold 70/15/15 per cent of
    records, not of calendar time. interpolation="lower" makes each boundary an
    observed timestamp; with `<=`, all records sharing a minute fall in one block.
    """
    cut_train = ts.quantile(q[0], interpolation="lower")
    cut_val = ts.quantile(q[1], interpolation="lower")
    return {"quantiles": list(q), "cut_train": str(cut_train), "cut_val": str(cut_val)}


def fix_boundaries(ts: pd.Series, path: Path) -> dict:
    """Write the boundaries on the first run; on every later run, verify them."""
    bounds = compute_boundaries(ts)
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != bounds:
            raise RuntimeError(f"Split boundaries differ from those recorded in {path}: "
                               f"{stored} vs {bounds}. Section 4.5 forbids revision; "
                               "investigate before deleting the file.")
        return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bounds, indent=2))
    return bounds


def assign_block(ts: pd.Series, bounds: dict) -> pd.Series:
    cut_train = pd.Timestamp(bounds["cut_train"])
    cut_val = pd.Timestamp(bounds["cut_val"])
    block = np.select([ts <= cut_train, ts <= cut_val], ["train", "val"], default="test")
    return pd.Series(pd.Categorical(block, categories=BLOCKS, ordered=True), index=ts.index)


def block_summary(df: pd.DataFrame) -> dict:
    g = df.groupby("block", observed=True)
    out = pd.DataFrame({
        "n_records": g.size(),
        "n_laundering": g["Is Laundering"].sum().astype("int64"),
        "first": g["Timestamp"].min().astype(str),
        "last": g["Timestamp"].max().astype(str),
    })
    out["share_records"] = (out["n_records"] / out["n_records"].sum()).round(4)
    out["ratio_1_in"] = (out["n_records"] / out["n_laundering"]).round(1)
    return {b: {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for b, row in out.to_dict(orient="index").items()}
