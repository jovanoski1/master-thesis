import pandas as pd

from aml.features import base_attributes


def test_base_attributes():
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2022-09-01 13:05", "2022-09-03 00:00"]),
        "Amount Paid": [10.0, 5.0], "Amount Received": [10.0, 4.5],
        "Payment Currency": pd.Categorical(["US Dollar", "Euro"]),
        "Receiving Currency": pd.Categorical(["US Dollar", "US Dollar"]),
        "Payment Format": pd.Categorical(["ACH", "Wire"]),
        "From Bank": pd.Categorical(["10", "20"]),
        "To Bank": pd.Categorical(["10", "30"]),
    })
    a = base_attributes(df)
    assert a["currency_mismatch"].tolist() == [0, 1]
    assert a["cross_bank"].tolist() == [0, 1]
    assert a["hour"].tolist() == [13, 0]
    assert a["day_of_week"].tolist() == [3, 5]       # Thursday, Saturday
    assert len(a.columns) == 9                       # Table 4.2: about ten
