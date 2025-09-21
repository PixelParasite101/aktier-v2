import numpy as np
import pandas as pd

from make_rebased_windows import make_windows_for_ticker, order_columns


def _toy_feature_df():
    # simple linear price to make rebasing predictable
    n = 20
    g = pd.DataFrame(
        {
            "Ticker": ["TEST"] * n,
            "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "AdjClose": np.arange(n, dtype=float) + 1.0,
            "Close": np.arange(n, dtype=float) + 1.0,
            "MA_20": np.nan,  # not enough data; should be tolerated
        }
    )
    return g


def test_make_windows_rebased_center_equals_100():
    g = _toy_feature_df()
    before, after = 3, 2
    out = make_windows_for_ticker(g, before, after)
    assert not out.empty
    # Ensure the rebased value at offset 0 is 100
    rows_offset0 = out[out["Offset"] == 0]
    assert (np.isclose(rows_offset0["AdjClose_Rebased"], 100.0)).all()


def test_order_columns_includes_expected_sequence():
    g = _toy_feature_df()
    out = make_windows_for_ticker(g, 2, 1)
    out = order_columns(out)
    expected_start = ["Ticker", "RefDate", "Offset", "Date"]
    assert list(out.columns[:4]) == expected_start
