"""Steps 2-3: load, check, attach typologies, partition, persist.

Usage:  python -m aml.prepare --variant HI-Small
        python -m aml.prepare            (both variants, processed independently)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from . import config, load, patterns, split


def prepare(variant: str, raw_dir: Path = config.DATA_RAW,
            interim_dir: Path = config.DATA_INTERIM,
            results_dir: Path = config.RESULTS) -> dict:
    t0 = time.perf_counter()
    raw_dir, interim_dir, results_dir = map(Path, (raw_dir, interim_dir, results_dir))

    df = load.read_transactions(raw_dir / f"{variant}_Trans.csv")
    pat = patterns.parse_patterns(raw_dir / f"{variant}_Patterns.txt")
    df, pattern_report = patterns.attach_typology(df, pat)
    df = load.finalise_types(df)

    integrity = load.integrity_checks(df, variant)
    bounds = split.fix_boundaries(df["Timestamp"], results_dir / f"split_{variant}.json")
    df["block"] = split.assign_block(df["Timestamp"], bounds)

    interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_dir / f"{variant}.parquet", index=False)

    report = {
        "variant": variant,
        "integrity": integrity,
        "patterns": pattern_report,
        "split_boundaries": bounds,
        "blocks": split.block_summary(df),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"prepare_{variant}.json").write_text(
        json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=config.VARIANTS, action="append")
    variants = ap.parse_args().variant or list(config.VARIANTS)
    for v in variants:
        r = prepare(v)
        failed = [k for k, ok in r["integrity"]["checks"].items() if not ok]
        print(f"{v}: {r['integrity']['n_records']:,} records, "
              f"{r['integrity']['n_laundering']:,} laundering, "
              f"span {r['integrity']['span_days']} days, {r['runtime_seconds']} s")
        print(f"  failed checks: {failed or 'none'}  -> results/prepare_{v}.json")


if __name__ == "__main__":
    main()
