import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aml import aggregates as agg
from aml import fx, graph, screening


def toy_events(n=2500, n_acc=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.choice(np.arange(0, 3 * 1440, 9), size=n))     # many minute ties
    src, dst = rng.integers(0, n_acc, n), rng.integers(0, n_acc, n)
    amt = rng.lognormal(5, 2, n)
    return src.astype(np.int64), dst.astype(np.int64), t.astype(np.int64), amt, amt * 0.97


def naive_b(src, dst, t, ao, ai, w):
    """Independent reference: one boolean mask per transaction."""
    rows = []
    for r in range(len(t)):
        win = (t >= t[r] - w) & (t < t[r])
        o, i = win & (src == src[r]), win & (dst == dst[r])
        si, do = int((win & (dst == src[r])).sum()), int((win & (src == dst[r])).sum())
        lo, li = np.log1p(ao[o]), np.log1p(ai[i])

        def prev(a):
            ts = t[((src == a) | (dst == a)) & (t < t[r])]
            return (t[r] - ts.max()) / 60 if len(ts) else t[r] / 60
        rows.append({
            "src_out_cnt": o.sum(), "src_out_sum": ao[o].sum(),
            "src_out_mean_log": lo.mean() if len(lo) else 0,
            "src_out_std_log": lo.std(ddof=1) if len(lo) > 1 else 0,
            "src_out_n_cp": len(np.unique(dst[o])),
            "dst_in_cnt": i.sum(), "dst_in_sum": ai[i].sum(),
            "dst_in_mean_log": li.mean() if len(li) else 0,
            "dst_in_std_log": li.std(ddof=1) if len(li) > 1 else 0,
            "dst_in_n_cp": len(np.unique(src[i])),
            "src_in_share": (si + 1) / (si + o.sum() + 2),
            "dst_in_share": (i.sum() + 1) / (i.sum() + do + 2),
            "src_hours_since_prev": prev(src[r]), "dst_hours_since_prev": prev(dst[r])})
    return pd.DataFrame(rows)


def test_set_b_equals_naive_reference_with_ties():
    src, dst, t, ao, ai = toy_events()
    w = 1440
    got = agg.set_b(src, dst, t, ao, ai, w, n_accounts=30)
    ref = naive_b(src, dst, t, ao, ai, w)
    for c in ref.columns:
        np.testing.assert_allclose(got[c].to_numpy(), ref[c].to_numpy(), rtol=2e-5,
                                   atol=1e-5, err_msg=c)
    rep = agg.verify_b(src, dst, t, ao, ai, got, w, np.arange(0, len(t), 7))
    assert rep["passed"]


def test_window_boundaries():
    # account 0 pays 1 at t=0, 2 at t=60, 3 at t=100; queries at the same minutes
    src = np.array([0, 0, 0]); dst = np.array([1, 2, 1]); t = np.array([0, 60, 100])
    ao = np.array([1.0, 2.0, 3.0])
    got = agg.set_b(src, dst, t, ao, ao, w=60, n_accounts=3)
    assert got["src_out_cnt"].tolist() == [0, 1, 1]       # t=60 sees t=0 (= t - w); t=100 sees 60 only
    assert got["src_out_sum"].tolist() == [0, 1, 2]
    assert got["src_out_n_cp"].tolist() == [0, 1, 1]


def test_node_stats_match_networkx():
    rng = np.random.default_rng(3)
    u, v = rng.integers(0, 60, 400), rng.integers(0, 60, 400)
    keep = u != v
    u, v = u[keep] + 1000, v[keep] + 1000                   # non-contiguous codes
    nodes, st = graph.node_stats(u, v, chunk_budget=50)      # tiny budget: many chunks
    G = nx.DiGraph(list(zip(u.tolist(), v.tolist())))
    U = G.to_undirected()
    pr = nx.pagerank(G, alpha=0.85)
    cc = nx.clustering(U)
    for i, node in enumerate(nodes.tolist()):
        assert st["in_deg"][i] == G.in_degree(node)
        assert st["out_deg"][i] == G.out_degree(node)
        assert st["n_cp"][i] == U.degree(node)
        two = len(nx.single_source_shortest_path_length(U, node, cutoff=2)) - 1
        assert st["two_hop"][i] == two
        assert st["clustering"][i] == pytest.approx(cc[node])
        outs, ins = set(G.successors(node)), set(G.predecessors(node))
        assert st["reciprocity"][i] == pytest.approx(len(outs & ins) / len(outs | ins))
        assert st["pagerank"][i] == pytest.approx(pr[node] * len(nodes))
        assert st["component_size"][i] == len(nx.node_connected_component(U, node))


def test_set_c_uses_only_earlier_snapshots():
    src, dst, t, _, _ = toy_events(n=1500)
    origin = pd.Timestamp("2022-09-01 06:30")                # not a midnight
    C, snaps, snap_of_row = graph.set_c(src, dst, t, origin, "1D", 1440, progress=lambda m: None)
    assert (snap_of_row[snap_of_row >= 0] <= t[snap_of_row >= 0]).all()
    assert snaps["edges_strictly_before"].all()
    first = int(snaps["snapshot_minute"].min())
    assert first == 17 * 60 + 30                             # next midnight after 06:30
    before = t < first
    assert (C.loc[before, "src_out_deg"] == 0).all() and (C.loc[before, "src_component_size"] == 1).all()
    rep = graph.verify_c(src, dst, t, C, snap_of_row, 1440, np.arange(len(t)))
    assert rep["passed"] and rep["n_rows_checked"] > 0


def test_fx_rates_through_a_chain():
    rows = []
    for _ in range(30):   # EUR<->USD at 1.1 USD per EUR, GBP->EUR at 1.2 EUR per GBP
        rows += [("Euro", "US Dollar", 100.0, 110.0), ("US Dollar", "Euro", 110.0, 100.0),
                 ("UK Pound", "Euro", 10.0, 12.0), ("US Dollar", "US Dollar", 5.0, 5.0)]
    df = pd.DataFrame(rows, columns=["Payment Currency", "Receiving Currency",
                                     "Amount Paid", "Amount Received"])
    for c in ["Payment Currency", "Receiving Currency"]:
        df[c] = df[c].astype("category")
    rates, diag, pairs = fx.implied_rates(df, "US Dollar")
    assert rates["US Dollar"] == pytest.approx(1.0)
    assert rates["Euro"] == pytest.approx(1.1)
    assert rates["UK Pound"] == pytest.approx(1.32)
    assert (diag["method"] == "implied rate").all()
    usd = fx.to_base(df["Amount Paid"], df["Payment Currency"], rates)
    assert usd[2] == pytest.approx(13.2)


def test_screening_drops_later_duplicates_only():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    X = pd.DataFrame({"A_x": a, "A_const": 1.0, "A_y": rng.normal(size=5000),
                      "B_dup": 2 * a + 1e-6 * rng.normal(size=5000), "C_z": rng.normal(size=5000)})
    kept, rep, _ = screening.screen(X, list(X.columns), 0.95)
    assert kept == ["A_x", "A_y", "C_z"]
    assert rep["constant"] == ["A_const"] and rep["redundant"]["B_dup"]["duplicates"] == "A_x"


def test_build_end_to_end(tmp_path):
    from aml import build_features, prepare
    from tests.synthetic import write_synthetic
    write_synthetic(tmp_path / "raw", n=6000, seed=1)
    prepare.prepare("HI-Small", tmp_path / "raw", tmp_path / "interim", tmp_path / "results")
    rep = build_features.build("HI-Small", tmp_path / "interim", tmp_path / "features",
                               tmp_path / "results", progress=lambda m: None)
    assert rep["verification"]["set_b"]["passed"] and rep["verification"]["set_c"]["passed"]
    sets = json.loads((tmp_path / "results" / "feature_sets_HI-Small.json").read_text())["sets"]
    assert set(sets["A"]) <= set(sets["B"]) <= set(sets["C"])
    X = pd.read_parquet(tmp_path / "features" / "HI-Small.parquet")
    assert len(X) == 6000 and "typology" not in X.columns
    assert not X[sets["C"]].isna().any().any()


def test_tail_flag_and_weekend():
    from aml.build_features import tail_flag
    days = pd.to_datetime(["2022-09-01"] * 1000 + ["2022-09-02"] * 900 + ["2022-09-03"] * 950
                          + ["2022-09-04"] * 12 + ["2022-09-05"] * 3)
    tail, last = tail_flag(pd.Series(days), 0.10)
    assert str(last.date()) == "2022-09-03" and tail.sum() == 15 and tail[-15:].all()
    from aml.features import base_attributes
    ts = pd.to_datetime(["2022-09-02 10:00", "2022-09-03 10:00", "2022-09-04 10:00"])
    df = pd.DataFrame({"Timestamp": ts, "Amount Paid": 1.0, "Amount Received": 1.0,
                       "Payment Currency": pd.Categorical(["US Dollar"] * 3),
                       "Receiving Currency": pd.Categorical(["US Dollar"] * 3),
                       "Payment Format": pd.Categorical(["ACH"] * 3),
                       "From Bank": pd.Categorical(["1"] * 3), "To Bank": pd.Categorical(["2"] * 3)})
    from aml.build_features import set_a
    A = set_a(df, pd.Series({"US Dollar": 1.0}), np.ones(3, bool), "weekend")
    assert A["weekend"].tolist() == [0, 1, 1] and "day_of_week" not in A
