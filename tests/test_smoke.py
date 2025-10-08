import pandas as pd
import numpy as np
from pathlib import Path

from src.compute_features import add_features, order_columns as order_feature_cols
from src.make_rebased_windows import make_windows_for_ticker, order_columns as order_rebased_cols


def test_offline_smoke_end_to_end(tmp_path: Path):
    # 1) Synthetic raw OHLCV for two tickers
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    def mk_series(start):
        # linear drift, enough points for MAs and RSI
        n = len(dates)
        return np.linspace(start, start + n - 1, n, dtype=float)

    raw = pd.DataFrame(
        {
            "Ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "Date": list(dates) + list(dates),
            "Open": list(mk_series(10)) + list(mk_series(20)),
            "High": list(mk_series(11)) + list(mk_series(21)),
            "Low": list(mk_series(9)) + list(mk_series(19)),
            "Close": list(mk_series(10.5)) + list(mk_series(20.5)),
            "AdjClose": list(mk_series(10.25)) + list(mk_series(20.25)),
            "Volume": list(np.arange(len(dates)) * 100) + list(np.arange(len(dates)) * 200),
        }
    )

    # 2) Compute features (use AdjClose)
    feats = add_features(raw, ma_windows=[5, 10], rsi_len=6, use_adj=True)
    feats = order_feature_cols(feats)

    # Basic invariants: non-empty, contains indicators
    assert not feats.empty
    for c in ["MA_5", "MA_10", "RSI_6"]:
        assert c in feats.columns

    # 3) Rebased windows per ticker using a small window to keep output small
    all_rebased = []
    for _, g in feats.groupby("Ticker", sort=False, observed=False):
        out = make_windows_for_ticker(g, before=5, after=3)
        assert not out.empty
        out = order_rebased_cols(out)
        # Offset 0 should be 100 for rebased columns where reference exists
        rows0 = out[out["Offset"] == 0]
        if "AdjClose_Rebased" in rows0.columns:
            assert (np.isclose(rows0["AdjClose_Rebased"], 100.0)).all()
        all_rebased.append(out)

    # 4) Combined non-empty
    combined = pd.concat(all_rebased, ignore_index=True)
    assert not combined.empty
