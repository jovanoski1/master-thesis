"""Feature construction (Section 4.6).

Only the transaction-level attributes of Set A are defined so far. Sets B and C are
added once their window and snapshot specification is fixed.
"""
import numpy as np
import pandas as pd


def _same_value(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality of two categoricals with different category sets,
    compared on shared codes rather than by materialising millions of strings."""
    cats = a.cat.categories.union(b.cat.categories)
    return (a.cat.set_categories(cats).cat.codes.to_numpy()
            == b.cat.set_categories(cats).cat.codes.to_numpy())


def base_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Set A: attributes available from the transaction record in isolation (Table 4.2).

    Returned untransformed; logarithms, encoding and scaling belong to preprocessing,
    whose parameters are estimated on the training block only.
    """
    ts = df["Timestamp"]
    return pd.DataFrame({
        "amount_paid": df["Amount Paid"].astype("float64"),
        "amount_received": df["Amount Received"].astype("float64"),
        "payment_currency": df["Payment Currency"],
        "receiving_currency": df["Receiving Currency"],
        "currency_mismatch": (~_same_value(df["Payment Currency"],
                                           df["Receiving Currency"])).astype("int8"),
        "payment_format": df["Payment Format"],
        "hour": ts.dt.hour.astype("int8"),
        "day_of_week": ts.dt.dayofweek.astype("int8"),
        "cross_bank": (~_same_value(df["From Bank"], df["To Bank"])).astype("int8"),
    }, index=df.index)
