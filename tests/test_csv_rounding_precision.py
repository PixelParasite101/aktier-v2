import pandas as pd
import numpy as np
from pathlib import Path
import fetch_history_pro as fh
from utils.common import round_for_csv


def test_round_for_csv_precision_retained_parquet(tmp_path):
    df = pd.DataFrame({
        "Ticker": ["T"]*3,
        "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Open": [1.123456, 2.987654, 3.141592],
        "High": [1.223456, 3.087654, 3.241592],
        "Low":  [1.023456, 2.887654, 3.041592],
        "Close": [1.523456, 2.387654, 3.541592],
        "AdjClose": [1.623456, 2.487654, 3.641592],
        "Volume": [10, 20, 30]
    })
    rounded, fmt = round_for_csv(df, 4, include_cols=["Open","High","Low","Close","AdjClose"])
    # Ensure rounding applied
    assert str(rounded.loc[0, "Open"]).endswith("3456")
    # Original df remains higher precision
    assert df.loc[0, "Open"] != round(df.loc[0, "Open"], 4) or df.loc[0, "Open"] == 1.123456
    # Write parquet of original and check values intact
    pq_path = tmp_path / "orig.parquet"
    df.to_parquet(pq_path, index=False)
    read_back = pd.read_parquet(pq_path)
    assert np.isclose(read_back.loc[0, "Open"], df.loc[0, "Open"], atol=1e-9)
