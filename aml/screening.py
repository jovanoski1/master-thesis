"""Feature screening on the training block (Section 4.6, screening rule).

The rule is fixed in advance and applied once, identically for every model family:
  1. constant features are dropped;
  2. features are visited in their declared order (Set A, then B, then C); a feature
     whose |Spearman rho| with an already kept feature exceeds the threshold is dropped.
Because earlier features always win, a Set B or C feature can be dropped for duplicating
a Set A feature, but never the reverse, so the three sets stay nested.
"""
import numpy as np
import pandas as pd


def spearman(X: pd.DataFrame) -> pd.DataFrame:
    R = np.array(X.rank(method="average").to_numpy(), dtype=np.float64)   # writable copy
    R -= R.mean(axis=0)
    sd = R.std(axis=0)
    R /= np.where(sd > 0, sd, 1.0)
    return pd.DataFrame(R.T @ R / len(R), index=X.columns, columns=X.columns)


def screen(X: pd.DataFrame, order: list, threshold: float) -> tuple[list, dict, pd.DataFrame]:
    constant = [c for c in order if X[c].nunique(dropna=False) <= 1]
    candidates = [c for c in order if c not in constant]
    rho = spearman(X[candidates])
    kept, redundant = [], {}
    for c in candidates:
        if kept:
            r = rho.loc[c, kept].abs()
            if r.max() > threshold:
                redundant[c] = {"duplicates": r.idxmax(), "abs_rho": round(float(r.max()), 4)}
                continue
        kept.append(c)
    report = {"threshold": threshold, "n_rows": int(len(X)), "n_candidates": len(order),
              "constant": constant, "redundant": redundant, "n_kept": len(kept)}
    return kept, report, rho
