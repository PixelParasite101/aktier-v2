import pandas as pd
import os
import pytest
from pathlib import Path

from src import fetch_history_pro as fh


def test_save_prices_parquet_error(monkeypatch, tmp_path: Path):
    # Create a small dataframe
    df = pd.DataFrame({
        "Ticker": ["T"],
        "Date": [pd.Timestamp("2024-01-01")],
        "Open": [1.0],
        "High": [1.1],
        "Low": [0.9],
        "Close": [1.05],
        "AdjClose": [1.02],
        "Volume": [100],
    })
    out_dir = str(tmp_path)

    # Monkeypatch DataFrame.to_parquet to raise
    def fake_to_parquet(self, *args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    # Should not raise; errors are logged
    fh.save_prices(df, out_dir, per_ticker=True, compression="snappy", partition_by=None, float_dp=None)

    # CSV should still be written
    assert os.path.exists(os.path.join(out_dir, "T.csv"))


def test_incremental_merge_overlap():
    old = pd.DataFrame({
        "Ticker": ["A","A"],
        "Date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "Close": [1.0, 2.0],
    })
    new = pd.DataFrame({
        "Ticker": ["A","A"],
        "Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        "Close": [2.1, 3.0],
    })
    merged = fh.incremental_merge(old, new)
    # Should deduplicate on (Ticker,Date) keeping last occurrence
    assert len(merged) == 3
    # Date 2024-01-02 should reflect new value 2.1
    row = merged[merged["Date"]==pd.Timestamp("2024-01-02")].iloc[0]
    assert row["Close"] == 2.1
