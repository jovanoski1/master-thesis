"""Set B: per-account aggregates over a look-back window (Section 4.6).

Every aggregate for a transaction at minute t uses only events at minutes t' with
t - W <= t' < t. Events in the same minute as the transaction are excluded, including
other transactions of that minute, so ties in the minute-resolution timestamps cannot
leak information. All windows are evaluated exactly, without loops over transactions:
events are sorted by (account, minute), window bounds are found by binary search, and
sums come from prefix sums that restart at every account (so large totals of other
accounts cannot erode precision).
"""
import numpy as np
import pandas as pd

K = np.int64(1) << np.int64(32)          # key = account * K + minute
MINUTE = pd.Timedelta("1min")


def minutes(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((ts - origin) // MINUTE).to_numpy(dtype=np.int64)


def window_minutes(window: str) -> int:
    return int(pd.Timedelta(window) / MINUTE)


def account_codes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Codes for source and destination accounts on one shared index."""
    cats = df["src_account"].cat.categories.union(df["dst_account"].cat.categories)
    src = df["src_account"].cat.set_categories(cats).cat.codes.to_numpy(np.int64)
    dst = df["dst_account"].cat.set_categories(cats).cat.codes.to_numpy(np.int64)
    return src, dst, len(cats)


class EventIndex:
    """Events of one kind sorted by (account, minute), with per-account prefix sums."""

    def __init__(self, acct: np.ndarray, t: np.ndarray, values: dict | None = None):
        key = acct * K + t
        order = np.argsort(key, kind="stable")
        self.key, self.t = key[order], t[order]
        acct_sorted = acct[order]
        self.prefix = {name: pd.Series(np.asarray(v, dtype=np.float64)[order])
                       .groupby(acct_sorted, sort=False).cumsum().to_numpy()
                       for name, v in (values or {}).items()}

    def bounds(self, a: np.ndarray, t: np.ndarray, w: int | None):
        start = np.searchsorted(self.key, a * K, "left")
        hi = np.searchsorted(self.key, a * K + t, "left")              # t' < t
        lo = start if w is None else np.maximum(
            np.searchsorted(self.key, a * K + (t - w), "left"), start)  # t' >= t - w
        return start, lo, hi

    def window(self, a, t, w):
        start, lo, hi = self.bounds(a, t, w)

        def wsum(name):
            p = self.prefix[name]
            upper = np.where(hi > start, p[np.maximum(hi - 1, 0)], 0.0)
            lower = np.where(lo > start, p[np.maximum(lo - 1, 0)], 0.0)
            return upper - lower
        return hi - lo, wsum

    def previous_minute(self, a, t):
        start, _, hi = self.bounds(a, t, None)
        return np.where(hi > start, self.t[np.maximum(hi - 1, 0)], -1)


def distinct_counterparties(acct, cp, t, q_acct, q_t, w, n_accounts):
    """Number of distinct counterparties of q_acct with an event in [q_t - w, q_t).

    For one (account, counterparty) pair, the indicator is 1 exactly for query minutes in
    the union of the intervals (e, e + w] over its event minutes e. Overlapping intervals
    are merged per pair; the count for a query is then (#merged intervals starting before
    q_t) - (#merged intervals ending before q_t), both found by binary search.
    """
    pair = acct * np.int64(n_accounts) + cp
    order = np.lexsort((t, pair))
    p, tt, a = pair[order], t[order], acct[order]
    new = np.ones(len(p), dtype=bool)
    new[1:] = (p[1:] != p[:-1]) | (tt[1:] > tt[:-1] + w)
    first = np.flatnonzero(new)
    last = np.append(first[1:] - 1, len(p) - 1)
    S = np.sort(a[first] * K + tt[first])
    F = np.sort(a[first] * K + tt[last] + w)
    base = q_acct * K
    n_start = np.searchsorted(S, base + q_t, "left") - np.searchsorted(S, base, "left")
    n_end = np.searchsorted(F, base + q_t, "left") - np.searchsorted(F, base, "left")
    return n_start - n_end


def _moments(idx: EventIndex, a, t, w):
    cnt, wsum = idx.window(a, t, w)
    total, s1, s2 = wsum("amt"), wsum("lg"), wsum("lg2")
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 1, (s2 - cnt * mean ** 2) / np.maximum(cnt - 1, 1), 0.0)
    return cnt, total, mean, np.sqrt(np.clip(var, 0.0, None))


def set_b(src, dst, t, amt_out, amt_in, w: int, n_accounts: int) -> pd.DataFrame:
    """Fourteen Set B features. Amounts are in the base currency; means and standard
    deviations are taken of log(1 + amount), which is numerically stable and robust to
    amounts that span many orders of magnitude."""
    lg_out, lg_in = np.log1p(amt_out), np.log1p(amt_in)
    out = EventIndex(src, t, {"amt": amt_out, "lg": lg_out, "lg2": lg_out ** 2})
    inc = EventIndex(dst, t, {"amt": amt_in, "lg": lg_in, "lg2": lg_in ** 2})
    every = EventIndex(np.concatenate([src, dst]), np.concatenate([t, t]))

    so_cnt, so_sum, so_mean, so_std = _moments(out, src, t, w)
    di_cnt, di_sum, di_mean, di_std = _moments(inc, dst, t, w)
    si_cnt = inc.window(src, t, w)[0]
    do_cnt = out.window(dst, t, w)[0]

    def hours_since_previous(a):
        prev = every.previous_minute(a, t)
        # no earlier activity: hours since the start of observation (a lower bound)
        return np.where(prev >= 0, t - prev, t) / 60.0

    feats = {
        "src_out_cnt": so_cnt, "src_out_sum": so_sum,
        "src_out_mean_log": so_mean, "src_out_std_log": so_std,
        "src_out_n_cp": distinct_counterparties(src, dst, t, src, t, w, n_accounts),
        "dst_in_cnt": di_cnt, "dst_in_sum": di_sum,
        "dst_in_mean_log": di_mean, "dst_in_std_log": di_std,
        "dst_in_n_cp": distinct_counterparties(dst, src, t, dst, t, w, n_accounts),
        "src_in_share": (si_cnt + 1) / (si_cnt + so_cnt + 2),
        "dst_in_share": (di_cnt + 1) / (di_cnt + do_cnt + 2),
        "src_hours_since_prev": hours_since_previous(src),
        "dst_hours_since_prev": hours_since_previous(dst),
    }
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in feats.items()})


def verify_b(src, dst, t, amt_out, amt_in, B: pd.DataFrame, w: int, rows: np.ndarray) -> dict:
    """Recompute Set B for the given rows by direct filtering of each account's events
    and compare with the vectorised result (replaces Listing 4.4)."""
    by_src = np.argsort(src, kind="stable")
    by_dst = np.argsort(dst, kind="stable")
    src_s, dst_s = src[by_src], dst[by_dst]

    def events(sorted_codes, order, a):
        i, j = np.searchsorted(sorted_codes, a, "left"), np.searchsorted(sorted_codes, a, "right")
        return order[i:j]

    got_all = {c: B[c].to_numpy() for c in B.columns}
    worst = {c: 0.0 for c in B.columns}
    for r in rows:
        a, b, tr = src[r], dst[r], t[r]
        e_out = events(src_s, by_src, a)
        e_out = e_out[(t[e_out] >= tr - w) & (t[e_out] < tr)]
        e_in = events(dst_s, by_dst, b)
        e_in = e_in[(t[e_in] >= tr - w) & (t[e_in] < tr)]
        e_src_in = events(dst_s, by_dst, a)
        e_dst_out = events(src_s, by_src, b)
        n_si = int(((t[e_src_in] >= tr - w) & (t[e_src_in] < tr)).sum())
        n_do = int(((t[e_dst_out] >= tr - w) & (t[e_dst_out] < tr)).sum())

        def mom(v):
            lg = np.log1p(v)
            return (len(v), v.sum(), lg.mean() if len(v) else 0.0,
                    lg.std(ddof=1) if len(v) > 1 else 0.0)

        def prev(a_):
            ts = np.concatenate([t[events(src_s, by_src, a_)], t[events(dst_s, by_dst, a_)]])
            ts = ts[ts < tr]
            return (tr - ts.max()) / 60.0 if len(ts) else tr / 60.0

        c1, s1, m1, d1 = mom(amt_out[e_out])
        c2, s2, m2, d2 = mom(amt_in[e_in])
        ref = {"src_out_cnt": c1, "src_out_sum": s1, "src_out_mean_log": m1,
               "src_out_std_log": d1, "src_out_n_cp": len(np.unique(dst[e_out])),
               "dst_in_cnt": c2, "dst_in_sum": s2, "dst_in_mean_log": m2,
               "dst_in_std_log": d2, "dst_in_n_cp": len(np.unique(src[e_in])),
               "src_in_share": (n_si + 1) / (n_si + c1 + 2),
               "dst_in_share": (c2 + 1) / (c2 + n_do + 2),
               "src_hours_since_prev": prev(a), "dst_hours_since_prev": prev(b)}
        for c, v in ref.items():
            got = float(got_all[c][r])
            scale = max(1.0, abs(v))
            worst[c] = max(worst[c], abs(got - v) / scale)
    # float32 storage: relative error up to ~1e-6 is rounding, not a discrepancy
    return {"n_rows": int(len(rows)), "max_relative_error": worst,
            "passed": bool(max(worst.values()) < 1e-4)}
