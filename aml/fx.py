"""Conversion of amounts into one currency (Section 4.6).

AMLworld records each amount in its own currency, so raw amounts are not comparable
across currencies. A cross-currency transaction reveals the exchange rate between its two
currencies (amount received / amount paid). The median rate per currency pair, estimated
on the training block only, links every currency to the base currency through the chain
of observed pairs.
"""
from collections import deque

import numpy as np
import pandas as pd

from .features import _same_value


def implied_rates(train: pd.DataFrame, base: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return base-currency units per unit of each currency, a per-currency diagnostics
    table, and the per-pair median log rates with their interquartile range."""
    currencies = sorted(set(train["Payment Currency"].astype(str).unique())
                        | set(train["Receiving Currency"].astype(str).unique()))
    cross = ~_same_value(train["Payment Currency"], train["Receiving Currency"])
    sub = train.loc[cross, ["Payment Currency", "Receiving Currency",
                            "Amount Paid", "Amount Received"]]
    sub = sub[(sub["Amount Paid"] > 0) & (sub["Amount Received"] > 0)]
    k = np.log(sub["Amount Received"].astype(float) / sub["Amount Paid"].astype(float))
    pairs = (pd.DataFrame({"pay": sub["Payment Currency"].astype(str).to_numpy(),
                           "rec": sub["Receiving Currency"].astype(str).to_numpy(),
                           "k": k.to_numpy()})
             .groupby(["pay", "rec"])["k"]
             .agg(n="size", log_rate="median",
                  iqr=lambda x: float(np.subtract(*np.percentile(x, [75, 25]))))
             .reset_index()
             .sort_values(["n", "pay", "rec"], ascending=[False, True, True]))

    # 1 unit of `pay` = exp(k) units of `rec`  =>  lu(pay) = k + lu(rec)
    log_usd = {base: 0.0} if base in currencies else {}
    if not log_usd and len(currencies) == 1:
        log_usd = {currencies[0]: 0.0}
    queue = deque(log_usd)
    while queue:
        c = queue.popleft()
        for p, r, kk in pairs[["pay", "rec", "log_rate"]].itertuples(index=False):
            if p == c and r not in log_usd:
                log_usd[r] = log_usd[c] - kk
                queue.append(r)
            elif r == c and p not in log_usd:
                log_usd[p] = kk + log_usd[c]
                queue.append(p)

    rates = pd.Series({c: float(np.exp(v)) for c, v in log_usd.items()}, dtype=float)
    method = {c: "implied rate" for c in rates.index}
    missing = [c for c in currencies if c not in rates.index]
    if missing:   # no chain of cross-currency pairs: scale-match medians instead (flagged)
        known = train["Payment Currency"].astype(str).isin(rates.index)
        ref = float(np.median(train.loc[known, "Amount Paid"].astype(float).to_numpy()
                              * train.loc[known, "Payment Currency"].astype(str)
                              .map(rates).to_numpy())) if known.any() else 1.0
        for c in missing:
            med = float(train.loc[train["Payment Currency"].astype(str) == c,
                                  "Amount Paid"].median())
            rates[c] = ref / med if med > 0 else np.nan
            method[c] = "median scaling (no linking pair)"
    diag = pd.DataFrame({"base_units_per_unit": rates, "method": pd.Series(method)})
    diag.index.name = "currency"
    return rates.sort_index(), diag.sort_index(), pairs.reset_index(drop=True)


def to_base(amount: pd.Series, currency: pd.Series, rates: pd.Series) -> np.ndarray:
    """Multiply each amount by the rate of its currency (via categorical codes)."""
    cats = currency.cat.categories.astype(str)
    r = np.append(rates.reindex(cats).to_numpy(dtype=float), np.nan)  # code -1 -> NaN
    return amount.to_numpy(dtype=float) * r[currency.cat.codes.to_numpy()]
