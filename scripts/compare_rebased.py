import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_rebased.py TICKER [rebased_dir]")
        sys.exit(1)
    ticker = sys.argv[1]
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("rebased")
    csv_path = base / f"{ticker}_rebased.csv"
    pq_path = base / f"{ticker}_rebased.parquet"
    if not csv_path.exists() or not pq_path.exists():
        print(f"Missing files for {ticker} in {base}")
        sys.exit(2)

    df_csv = pd.read_csv(csv_path, parse_dates=["RefDate", "Date"], dayfirst=False)
    df_pq = pd.read_parquet(pq_path)
    # Ensure datetime types for fair compare
    for c in ("RefDate", "Date"):
        if c in df_pq.columns:
            df_pq[c] = pd.to_datetime(df_pq[c], errors="coerce")

    # Sort and align
    key = [c for c in ["Ticker", "RefDate", "Offset", "Date"] if c in df_csv.columns and c in df_pq.columns]
    df_csv = df_csv.sort_values(key).reset_index(drop=True)
    df_pq = df_pq.sort_values(key).reset_index(drop=True)
    if len(df_csv) != len(df_pq):
        print(f"Row count differs: CSV={len(df_csv)} Parquet={len(df_pq)}")
    else:
        print(f"Row count: {len(df_csv)}")

    # Compare numeric columns
    common_cols = [c for c in df_csv.columns if c in df_pq.columns]
    num_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_csv[c]) or pd.api.types.is_numeric_dtype(df_pq[c])]

    def nan_count(s):
        return int(pd.isna(s).sum())

    print("Column, CSV_NaN, PQ_NaN, Diff_Count")
    for c in num_cols:
        csv_nan = nan_count(df_csv[c])
        pq_nan = nan_count(df_pq[c])
        # Compute difference counts ignoring rows where either side is NaN
        mask = ~(pd.isna(df_csv[c]) | pd.isna(df_pq[c]))
        diff = int(np.sum(~np.isclose(pd.to_numeric(df_csv.loc[mask, c], errors='coerce'),
                                     pd.to_numeric(df_pq.loc[mask, c], errors='coerce'),
                                     equal_nan=False)))
        print(f"{c}, {csv_nan}, {pq_nan}, {diff}")

    # If lengths equal, spot-check some rows
    if len(df_csv) == len(df_pq):
        mismatched_cols = []
        for c in num_cols:
            mask = ~(pd.isna(df_csv[c]) | pd.isna(df_pq[c]))
            if not np.allclose(pd.to_numeric(df_csv.loc[mask, c], errors='coerce'),
                               pd.to_numeric(df_pq.loc[mask, c], errors='coerce'), equal_nan=False):
                mismatched_cols.append(c)
        if mismatched_cols:
            print("Mismatched columns:", ", ".join(mismatched_cols))
        else:
            print("All numeric columns match where not NaN.")


if __name__ == "__main__":
    main()
