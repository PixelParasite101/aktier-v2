"""Build an index for rebased windows.

Produces a parquet index file with one row per (Ticker, RefDate) containing
summary statistics and a small downsampled signature for fast prefiltering.

Usage (PowerShell):
  & "./venv/Scripts/python.exe" "./scripts/build_rebased_index.py" --ref-dir rebased --out rebased_index.parquet
"""
from pathlib import Path
import argparse
from typing import List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build compact index for rebased windows")
    p.add_argument("--ref-dir", default="rebased", help="Folder with rebased parquet files")
    p.add_argument("--match-cols", default="AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased",
                   help="Comma-separated rebased columns to summarize")
    p.add_argument("--lookback", type=int, default=75, help="Lookback length L (use same as matcher)")
    p.add_argument("--downsample", type=int, default=8, help="Number of samples in signature per column")
    p.add_argument("--out", default="rebased_index.parquet", help="Output parquet path")
    return p.parse_args()


def summarize_series(x: np.ndarray) -> dict:
    # x: 1d numeric array (may contain NaN)
    if len(x) == 0 or np.isnan(x).all():
        return {"mean": np.nan, "std": np.nan, "slope": np.nan}
    # use only finite values for mean/std; slope computed on finite pairs
    finite_mask = np.isfinite(x)
    vals = x[finite_mask]
    mean = float(np.nanmean(vals)) if vals.size else float("nan")
    std = float(np.nanstd(vals, ddof=0)) if vals.size else float("nan")
    slope = float("nan")
    if vals.size >= 2:
        # approximate slope vs index (not offset) to capture trend
        idx = np.arange(len(x))[finite_mask]
        try:
            a, b = np.polyfit(idx, vals, 1)
            slope = float(a)
        except Exception:
            slope = float("nan")
    return {"mean": mean, "std": std, "slope": slope}


def downsample_series(x: np.ndarray, n: int) -> List[float]:
    if n <= 0:
        return []
    if len(x) == 0:
        return [float("nan")] * n
    # linear sample indices across whole array (preserve ends)
    idx = np.linspace(0, max(0, len(x) - 1), num=n, dtype=int)
    out = []
    for i in idx:
        v = x[i]
        out.append(float(v) if np.isfinite(v) else float("nan"))
    return out


def build_index(ref_dir: Path, match_cols: List[str], lookback: int, downsample: int):
    files = sorted([p for p in ref_dir.glob("*.parquet") if p.is_file()])
    rows = []
    lb_start = -lookback + 1
    lb_end = 0
    for fp in files:
        # attempt to inspect parquet schema once (pyarrow) and then read only needed columns
        cols = ["Ticker", "RefDate", "Offset"] + match_cols
        cols_present = None
        try:
            import pyarrow.parquet as pq

            try:
                pqf = pq.ParquetFile(str(fp))
                pa_cols = [c for c in pqf.schema.names]
                cols_present = [c for c in cols if c in pa_cols]
            except Exception:
                cols_present = None
        except Exception:
            # pyarrow not available or failed; will fallback to pandas-only read
            cols_present = None

        if cols_present:
            try:
                df = pd.read_parquet(fp, columns=cols_present)
            except Exception:
                # fallback to full read
                df = pd.read_parquet(fp)
                cols_present = [c for c in df.columns if c in cols]
                df = df[[c for c in cols_present]]
        else:
            # fallback: read whole file once and subselect
            df = pd.read_parquet(fp)
            cols_present = [c for c in df.columns if c in cols]
            df = df[[c for c in cols_present]]

        # keep only offsets in lookback window
        df_lb = df[(df["Offset"] >= lb_start) & (df["Offset"] <= lb_end)]
        if df_lb.empty:
            continue

        for (ticker, refdate), g in df_lb.groupby(["Ticker", "RefDate"], sort=False):
            g = g.sort_values("Offset")
            row = {"Ticker": str(ticker), "RefDate": pd.to_datetime(refdate), "SourceFile": fp.name}
            for col in match_cols:
                if col in g.columns:
                    arr = pd.to_numeric(g[col], errors="coerce").to_numpy()
                else:
                    arr = np.array([])
                stats = summarize_series(arr)
                row[f"{col}_mean"] = stats["mean"]
                row[f"{col}_std"] = stats["std"]
                row[f"{col}_slope"] = stats["slope"]
                sig = downsample_series(arr, downsample)
                # store signature as native Python list; pyarrow will write this as a list/array column in Parquet
                row[f"{col}_sig"] = sig
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df_idx = pd.DataFrame(rows)
    # ensure types
    df_idx["RefDate"] = pd.to_datetime(df_idx["RefDate"], errors="coerce")
    return df_idx


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dir)
    match_cols = [c.strip() for c in args.match_cols.split(",") if c.strip()]
    print(f"Scanning {ref_dir} for match_cols={match_cols} lookback={args.lookback} downsample={args.downsample}")
    idx = build_index(ref_dir, match_cols, args.lookback, args.downsample)
    if idx.empty:
        print("No index rows generated")
        return
    outp = Path(args.out)
    idx.to_parquet(outp, index=False)
    print(f"Wrote index to {outp} ({len(idx)} rows)")


if __name__ == "__main__":
    main()
