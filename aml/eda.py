"""Temporal EDA that must precede the freezing of Sections 4.6 (Sets B and C).

It answers three questions on which the feature specification depends:
  1. How long is the observed period, and is laundering evenly spread over it?
  2. Does a rolling window of a given length ever bind, or is it cumulative in practice?
  3. With snapshots at a given frequency, how many transactions have no snapshot at all?

Usage:  python -m aml.eda --variant HI-Small
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

WINDOWS = ["1D", "7D", "30D"]
SNAPSHOT_FREQS = ["12h", "1D", "7D"]


def window_truncation(ts: pd.Series, start: pd.Timestamp) -> dict:
    """Share of transactions whose look-back window reaches before the first record,
    i.e. for which the 'rolling' aggregate is in fact an expanding one."""
    return {w: round(float((ts - pd.Timedelta(w) < start).mean()), 4) for w in WINDOWS}


def snapshot_coverage(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """A transaction at t inherits the latest snapshot s <= t, built from records < s.
    Reports, per block, the share of transactions that precede the first snapshot."""
    out = {}
    for f in SNAPSHOT_FREQS:
        first = start.floor("D") + pd.Timedelta(f)
        n_snap = len(pd.date_range(first, end, freq=f))
        no_snap = df["Timestamp"] < first
        out[f] = {"n_snapshots": n_snap,
                  "share_without_snapshot": {
                      b: round(float(no_snap[df["block"] == b].mean()), 4)
                      for b in df["block"].cat.categories}}
    return out


def history_availability(df: pd.DataFrame) -> dict:
    """Share of transactions, per block, whose account was active strictly earlier."""
    parts = []
    for role in ["src_account", "dst_account"]:
        first = df.groupby(role, observed=True)["Timestamp"].min()
        first.index = first.index.astype(str)   # the two roles have different categories
        parts.append(first)
    first_seen = pd.concat(parts).groupby(level=0).min()
    out = {}
    for role in ["src_account", "dst_account"]:
        seen = df[role].astype(str).map(first_seen) < df["Timestamp"]
        out[role] = {b: round(float(seen[df["block"] == b].mean()), 4)
                     for b in df["block"].cat.categories}
    return out


def plot_timeline(df: pd.DataFrame, bounds: dict, path: Path, variant: str) -> None:
    hourly = (df["Is Laundering"].astype("int64").set_axis(df["Timestamp"])
              .resample("1h").agg(["size", "sum"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly["size"], lw=0.8, label="transactions per hour")
    ax.set_ylabel("transactions per hour")
    ax2 = ax.twinx()
    ax2.plot(hourly.index, hourly["sum"], lw=0.8, color="tab:red",
             label="laundering per hour")
    ax2.set_ylabel("laundering transactions per hour")
    for key, name in [("cut_train", "train | val"), ("cut_val", "val | test")]:
        ax.axvline(pd.Timestamp(bounds[key]), color="k", ls="--", lw=0.8)
        ax.text(pd.Timestamp(bounds[key]), ax.get_ylim()[1], name, ha="center",
                va="bottom", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"{variant}: activity over the observed period", pad=16)
    fig.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(variant: str, interim_dir: Path = config.DATA_INTERIM,
        results_dir: Path = config.RESULTS) -> dict:
    interim_dir, results_dir = Path(interim_dir), Path(results_dir)
    cols = ["Timestamp", "src_account", "dst_account", "Is Laundering", "block"]
    df = pd.read_parquet(interim_dir / f"{variant}.parquet", columns=cols)
    bounds = json.loads((results_dir / f"split_{variant}.json").read_text())
    start, end = df["Timestamp"].min(), df["Timestamp"].max()

    y = df["Is Laundering"].astype("int64")
    daily = y.groupby(df["Timestamp"].dt.floor("D")).agg(
        n_records="size", n_laundering="sum")
    daily["laundering_per_1000"] = (1000 * daily["n_laundering"] / daily["n_records"]).round(3)
    daily.to_csv(results_dir / f"eda_daily_{variant}.csv")

    report = {
        "variant": variant,
        "first": str(start), "last": str(end),
        "span_days": round((end - start) / pd.Timedelta(days=1), 3),
        "window_truncated_share": window_truncation(df["Timestamp"], start),
        "snapshot_coverage": snapshot_coverage(df, start, end),
        "account_seen_before_share": history_availability(df),
    }
    (results_dir / f"eda_{variant}.json").write_text(json.dumps(report, indent=2))
    plot_timeline(df, bounds, results_dir / f"eda_timeline_{variant}.png", variant)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=config.VARIANTS, action="append")
    for v in ap.parse_args().variant or list(config.VARIANTS):
        r = run(v)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
