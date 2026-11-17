import json

import pandas as pd
import pytest

from aml import config, load, patterns, prepare, split
from tests.synthetic import write_synthetic


@pytest.fixture
def raw(tmp_path):
    src = write_synthetic(tmp_path / "raw")
    return tmp_path, src


def test_direction_preserved(raw):
    tmp, src = raw
    df = load.read_transactions(tmp / "raw" / "HI-Small_Trans.csv")
    assert (df["src_account"].values == src["src"].values).all()
    assert (df["dst_account"].values == src["dst"].values).all()


def test_unexpected_header_is_refused(raw):
    tmp, _ = raw
    p = tmp / "raw" / "HI-Small_Trans.csv"
    text = p.read_text().replace("From Bank,Account,To Bank", "To Bank,Account,From Bank", 1)
    p.write_text(text)
    with pytest.raises(ValueError):
        load.read_transactions(p)


def test_patterns_join_despite_bank_padding(raw):
    tmp, src = raw
    df = load.read_transactions(tmp / "raw" / "HI-Small_Trans.csv")
    pat = patterns.parse_patterns(tmp / "raw" / "HI-Small_Patterns.txt")
    out, rep = patterns.attach_typology(df, pat)
    assert rep["n_pattern_transactions_unmatched"] == 0
    assert rep["n_typology_rows_labelled_licit"] == 0
    assert rep["n_laundering_rows_without_typology"] == 0
    assert rep["unknown_typology_labels"] == []
    assert set(out["typology"].dropna()) == {"fan-out", "fan-in", "cycle", "stack"}
    assert "typology" in config.NON_FEATURE_COLS


def test_split_is_chronological_and_tie_consistent(raw):
    tmp, _ = raw
    df = load.finalise_types(load.read_transactions(tmp / "raw" / "HI-Small_Trans.csv"))
    b = split.fix_boundaries(df["Timestamp"], tmp / "results" / "split.json")
    df["block"] = split.assign_block(df["Timestamp"], b)
    codes = df["block"].cat.codes
    assert codes.is_monotonic_increasing                     # train < val < test in time
    assert df.groupby("Timestamp")["block"].nunique().max() == 1   # no minute is split
    share = df["block"].value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.02 and abs(share["test"] - 0.15) < 0.02


def test_boundaries_cannot_be_revised(raw):
    tmp, _ = raw
    df = load.finalise_types(load.read_transactions(tmp / "raw" / "HI-Small_Trans.csv"))
    path = tmp / "results" / "split.json"
    split.fix_boundaries(df["Timestamp"], path)
    split.fix_boundaries(df["Timestamp"], path)              # identical: accepted
    with pytest.raises(RuntimeError):
        split.fix_boundaries(df["Timestamp"] + pd.Timedelta("1h"), path)


def test_prepare_end_to_end(raw):
    tmp, _ = raw
    rep = prepare.prepare("HI-Small", tmp / "raw", tmp / "interim", tmp / "results")
    assert (tmp / "interim" / "HI-Small.parquet").exists()
    assert rep["integrity"]["checks"]["timestamps_parsed_and_ordered"]
    saved = json.loads((tmp / "results" / "prepare_HI-Small.json").read_text())
    assert set(saved["blocks"]) == {"train", "val", "test"}


def test_join_diagnostics_detect_inconsistencies(raw):
    tmp, src = raw
    p = tmp / "raw" / "HI-Small_Trans.csv"
    lines = p.read_text().splitlines()
    pos = [i for i, line in enumerate(lines[1:], start=1) if line.endswith(",1")]
    lines[pos[0]] = lines[pos[0]][:-1] + "0"      # pattern transaction labelled licit
    del lines[pos[1]]                              # pattern transaction absent from file
    p.write_text("\n".join(lines) + "\n")
    df = load.read_transactions(p)
    pat = patterns.parse_patterns(tmp / "raw" / "HI-Small_Patterns.txt")
    _, rep = patterns.attach_typology(df, pat)
    assert rep["n_typology_rows_labelled_licit"] == 1
    assert rep["n_pattern_transactions_unmatched"] == 1
