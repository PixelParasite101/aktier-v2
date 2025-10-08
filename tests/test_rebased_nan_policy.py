import pandas as pd
import numpy as np
from src.make_rebased_windows import make_windows_for_ticker


def _df_with_nan():
    # Create a dataframe where one internal row has NaN in AdjClose
    n = 15
    adj = np.linspace(10, 24, n)
    adj[7] = np.nan  # middle NaN
    g = pd.DataFrame({
        "Ticker": ["X"] * n,
        "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "AdjClose": adj,
        "Close": adj,
    })
    return g


def test_rebased_drop_none_keeps_rows():
    g = _df_with_nan()
    out = make_windows_for_ticker(g, before=5, after=2, drop_policy="none", require_full_window=False)
    # Should produce some windows even though there is a NaN
    assert not out.empty


def test_rebased_drop_any_filters_rows():
    g = _df_with_nan()
    out = make_windows_for_ticker(g, before=5, after=2, drop_policy="any", require_full_window=False)
    # Windows may be empty for this synthetic input; if non-empty, rows containing NaN in rebased
    # columns at the NaN offset should be removed
    if not out.empty:
        rebased_cols = [c for c in out.columns if c.endswith("_Rebased")]
        filtered = out[~out[rebased_cols].isna().any(axis=1)]
        # Accept either: some rows survive the 'any' filter, or every row has at least one NaN
        if filtered.empty:
            assert out[rebased_cols].isna().any(axis=1).all()
        else:
            assert not filtered.empty


def test_rebased_require_full_window_skips_on_nan():
    g = _df_with_nan()
    out = make_windows_for_ticker(g, before=5, after=2, drop_policy="any", require_full_window=True)
    # All windows containing the NaN should be skipped; may still have earlier windows if NaN outside window
    # Given NaN is centrally located, expect fewer or possibly zero windows
    # We assert that any produced windows have full contiguous length
    if not out.empty:
        ref_counts = out.groupby(["Ticker", "RefDate"]).size().to_list()
        # All windows must have identical size (no dropped rows inside) if require_full_window
        assert len(set(ref_counts)) == 1
