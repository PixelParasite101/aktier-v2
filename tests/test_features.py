import pandas as pd
import numpy as np

from compute_features import rsi_wilder, add_features, order_columns


def test_rsi_wilder_all_gains_is_100():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)
    rsi = rsi_wilder(s, length=3)
    # After warmup, RSI should be 100 where there are no losses
    assert np.isclose(rsi.iloc[-1], 100.0, equal_nan=False)


def test_rsi_wilder_all_losses_is_0():
    s = pd.Series([7, 6, 5, 4, 3, 2, 1], dtype=float)
    rsi = rsi_wilder(s, length=3)
    # After warmup, RSI should be 0 where there are no gains
    assert np.isclose(rsi.iloc[-1], 0.0, atol=1e-6, equal_nan=False)


def test_add_features_and_order_columns():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 10,
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Open": np.arange(10) + 10.0,
            "High": np.arange(10) + 11.0,
            "Low": np.arange(10) + 9.0,
            "Close": np.arange(10) + 10.5,
            "AdjClose": np.arange(10) + 10.25,
            "Volume": np.arange(10) * 100,
        }
    )
    out = add_features(df, ma_windows=[3, 5], rsi_len=3, use_adj=True)
    out = order_columns(out)
    # Columns appear in expected order
    first_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for i, c in enumerate(first_cols):
        assert out.columns[i] == c
    # Indicators present
    for c in ["MA_3", "MA_5", "RSI_3"]:
        assert c in out.columns
