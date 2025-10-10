import pandas as pd
import numpy as np
from pathlib import Path
import json

from utils.common import sample_and_round_df_to_csv, sample_and_round_parquet_to_csv


def test_sample_and_round_df_to_csv(tmp_path: Path):
    # small df (<= head+tail) -> full write
    df = pd.DataFrame({
        "Ticker": ["A"] * 3,
        "v0": [1.123456, 2.234567, 3.345678],
        "label": ["x", "y", "z"],
    })
    out = tmp_path / "out.csv"
    sample_and_round_df_to_csv(df, out, head=2, tail=2, float_dp=4)
    txt = out.read_text()
    assert "1.1235" in txt
    assert "3.3457" in txt


def test_sample_and_round_parquet_to_csv(tmp_path: Path):
    # large df (> head+tail) -> head+tail
    n = 2500
    df = pd.DataFrame({
        "Ticker": ["T"] * n,
        "v0": np.linspace(0, n - 1, n, dtype=float),
    })
    pq = tmp_path / "big.parquet"
    df.to_parquet(pq)
    out = tmp_path / "big_sampled.csv"
    sample_and_round_parquet_to_csv(pq, out, head=1000, tail=1000, float_dp=4)
    # Read back and assert we have 2000 rows
    df2 = pd.read_csv(out)
    assert len(df2) == 2000
    # rounding check
    assert df2['v0'].astype(float).map(lambda x: round(x, 4)).notnull().all()
