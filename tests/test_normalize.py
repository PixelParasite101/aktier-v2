import pandas as pd
from src.fetch_history_pro import normalize_prices


def test_normalize_prices_empty_returns_schema():
    df = normalize_prices(pd.DataFrame(), "AAPL")
    assert list(df.columns) == [
        "Ticker","Date","Open","High","Low","Close","AdjClose","Volume"
    ]
    assert df.empty


def test_normalize_prices_basic_mapping():
    raw = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Open": [1.2345, 2.3456],
        "High": [1.5, 2.6],
        "Low": [1.1, 2.2],
        "Close": [1.3, 2.4],
        "Adj Close": [1.25, 2.35],
        "Volume": [100, 200],
    }).set_index("Date")

    out = normalize_prices(raw, "AAPL")

    assert (out["Ticker"] == "AAPL").all()
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])  # naive UTC-normaliseret
    assert {"Open","High","Low","Close","AdjClose","Volume"}.issubset(out.columns)
    assert len(out) == 2
    # Adj Close er mappet til AdjClose og er numerisk
    assert out["AdjClose"].iloc[0] == 1.25
