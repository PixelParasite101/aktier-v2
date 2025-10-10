"""Round numeric columns in a flat vectors CSV to 4 decimals.

Usage (PowerShell):
  & .\venv\Scripts\python.exe .\scripts\round_flat_vectors_csv.py data\flat_vectors\all_vectors.csv

The script reads the CSV (or corresponding .parquet if present), rounds numeric columns to 4 decimals
and overwrites the CSV.
"""
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is on sys.path so `utils` package can be imported when
# running the script directly (not as a module). Project root is parent of repo
# (scripts/ located under repo root).
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from utils.common import round_for_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: round_flat_vectors_csv.py <csv-path>")
        raise SystemExit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        raise SystemExit(2)

    # Prefer parquet if available next to the csv
    pq = p.with_suffix('.parquet')
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(p)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_rounded, fmt = round_for_csv(df, 4, include_cols=numeric_cols)
    # Overwrite CSV
    df_rounded.to_csv(p, index=False, float_format=fmt)
    print(f"Wrote rounded CSV to {p} (rounded {len(numeric_cols)} numeric columns to 4 dp) ")


if __name__ == '__main__':
    main()
