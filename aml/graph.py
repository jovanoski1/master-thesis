"""Set C: structural statistics of the account graph from daily snapshots (Section 4.6).

Snapshot k is taken at minute s_k (a midnight) and contains the directed edges of all
transactions with s_k - window <= t < s_k; self-transfers are excluded. A transaction at
minute t inherits the most recent snapshot with s_k <= t, so every edge it can see is
strictly earlier than the transaction. Transactions before the first snapshot fall in the
burn-in period and receive the values of an isolated account.

Degree, counterparty count, reciprocity, the two-hop neighbourhood and the clustering
coefficient are computed with SciPy sparse matrices (row-chunked, so hubs cannot exhaust
memory); PageRank and weakly connected components with NetworkX.
"""
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROLES = ("src", "dst")
STATS = ("in_deg", "out_deg", "n_cp", "two_hop", "clustering", "reciprocity",
         "pagerank", "component_size")
ISOLATED = {"in_deg": 0.0, "out_deg": 0.0, "n_cp": 0.0, "two_hop": 0.0,
            "clustering": 0.0, "reciprocity": 0.0, "pagerank": 0.0, "component_size": 1.0}


def snapshot_minutes(origin: pd.Timestamp, t_max: int, freq: str) -> np.ndarray:
    """Snapshot times in minutes after `origin`, anchored at midnights."""
    step = int(pd.Timedelta(freq) / pd.Timedelta("1min"))
    first = int((origin.floor("D") + pd.Timedelta("1D") - origin) / pd.Timedelta("1min"))
    return np.arange(first, t_max + 1, step, dtype=np.int64)


def node_stats(u: np.ndarray, v: np.ndarray, chunk_budget: int = 20_000_000) -> tuple:
    """Statistics of every node of the directed graph with edges u -> v.

    Returns (nodes, stats) with stats[name][i] the value for nodes[i]. Repeated edges
    count once; self-loops must already be removed.
    """
    n_all = np.int64(max(u.max(initial=0), v.max(initial=0)) + 1)
    keys = np.unique(u * n_all + v)
    uu, vv = keys // n_all, keys % n_all
    nodes = np.unique(np.concatenate([uu, vv]))
    iu, iv = np.searchsorted(nodes, uu), np.searchsorted(nodes, vv)
    n = len(nodes)

    out_deg = np.bincount(iu, minlength=n).astype(np.float64)
    in_deg = np.bincount(iv, minlength=n).astype(np.float64)
    A = sp.csr_matrix((np.ones(2 * len(iu)), (np.concatenate([iu, iv]),
                                              np.concatenate([iv, iu]))), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    deg = np.diff(A.indptr).astype(np.float64)        # distinct counterparties

    reverse_exists = np.isin(vv * n_all + uu, keys)
    reciprocated = np.bincount(iu[reverse_exists], minlength=n)
    reciprocity = np.divide(reciprocated, deg, out=np.zeros(n), where=deg > 0)

    # two-hop neighbourhood and triangles, in row chunks bounded by the walk count
    walks = A @ deg
    cost = np.cumsum(walks + deg + 1.0)
    cuts = np.searchsorted(cost, np.arange(1, int(cost[-1] // chunk_budget) + 1) * chunk_budget)
    bounds = np.unique(np.concatenate([[0], np.minimum(cuts + 1, n), [n]]))
    two_hop, tri = np.zeros(n), np.zeros(n)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        R = A[start:stop]
        P = R @ A                                       # length-2 walks
        self_hit = (deg[start:stop] > 0).astype(np.float64)   # walk v -> w -> v
        two_hop[start:stop] = np.diff((P + R).indptr) - self_hit
        tri[start:stop] = np.asarray(P.multiply(R).sum(axis=1)).ravel() / 2.0
    clustering = np.divide(2.0 * tri, deg * (deg - 1), out=np.zeros(n), where=deg >= 2)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(iu.tolist(), iv.tolist()))
    pr = nx.pagerank(G, alpha=0.85)
    pagerank = np.fromiter((pr[i] for i in range(n)), dtype=np.float64, count=n) * n
    component = np.zeros(n)
    for comp in nx.weakly_connected_components(G):
        idx = np.fromiter(comp, dtype=np.int64)
        component[idx] = len(idx)

    stats = {"in_deg": in_deg, "out_deg": out_deg, "n_cp": deg, "two_hop": two_hop,
             "clustering": clustering, "reciprocity": reciprocity,
             "pagerank": pagerank, "component_size": component}
    return nodes, stats


def set_c(src, dst, t, origin: pd.Timestamp, freq: str, window: int,
          progress=print) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Sixteen Set C features (eight statistics for each of the two accounts), the
    per-snapshot report, and the snapshot minute assigned to every row (-1: none)."""
    snaps = snapshot_minutes(origin, int(t.max()), freq)
    assign = np.searchsorted(snaps, t, side="right") - 1
    out = {f"{r}_{s}": np.full(len(t), ISOLATED[s], dtype=np.float32)
           for r in ROLES for s in STATS}
    report = []
    for k, s in enumerate(snaps):
        rows = np.flatnonzero(assign == k)
        if len(rows) == 0:
            continue
        t0 = time.perf_counter()
        e = (t >= s - window) & (t < s) & (src != dst)
        if e.any():
            nodes, stats = node_stats(src[e], dst[e])
            for role, codes in zip(ROLES, (src[rows], dst[rows])):
                pos = np.searchsorted(nodes, codes)
                hit = pos < len(nodes)
                hit[hit] = nodes[pos[hit]] == codes[hit]
                for name in STATS:
                    out[f"{role}_{name}"][rows[hit]] = stats[name][pos[hit]]
            n_nodes, n_edges = len(nodes), int(e.sum())
            max_edge = int(t[e].max())
        else:
            n_nodes, n_edges, max_edge = 0, 0, -1
        secs = time.perf_counter() - t0
        report.append({"snapshot_minute": int(s), "n_edges": n_edges, "n_nodes": n_nodes,
                       "n_rows_served": int(len(rows)), "max_edge_minute": max_edge,
                       "edges_strictly_before": bool(max_edge < s), "seconds": round(secs, 1)})
        progress(f"  snapshot {k + 1}/{len(snaps)}: {n_edges:,} edges, {n_nodes:,} nodes, "
                 f"{len(rows):,} rows, {secs:.1f} s")
    snap_of_row = np.where(assign >= 0, snaps[np.maximum(assign, 0)], -1)
    return pd.DataFrame(out), pd.DataFrame(report), snap_of_row


def verify_c(src, dst, t, C: pd.DataFrame, snap_of_row, window: int, rows) -> dict:
    """For the given rows, recompute degree and counterparty counts directly from the raw
    transactions of the snapshot window, and confirm the snapshot precedes the row."""
    bad, checked = [], 0
    cols = {c: C[c].to_numpy() for c in ["src_out_deg", "src_in_deg", "src_n_cp",
                                         "dst_in_deg", "dst_out_deg", "dst_n_cp"]}
    rows = np.asarray(rows)
    for s in np.unique(snap_of_row[rows]):
        if s < 0:
            continue
        e = (t >= s - window) & (t < s) & (src != dst)
        es, ed = src[e], dst[e]
        for r in rows[snap_of_row[rows] == s]:
            checked += 1
            if s > t[r]:
                bad.append(("snapshot after row", int(r)))
                continue
            for role, a in (("src", src[r]), ("dst", dst[r])):
                outs, ins = np.unique(ed[es == a]), np.unique(es[ed == a])
                ref = {"out_deg": len(outs), "in_deg": len(ins),
                       "n_cp": len(np.union1d(outs, ins))}
                for name, val in ref.items():
                    if cols[f"{role}_{name}"][r] != val:
                        bad.append((f"{role}_{name}", int(r)))
    return {"n_rows_checked": checked, "n_mismatches": len(bad),
            "examples": bad[:5], "passed": not bad}
