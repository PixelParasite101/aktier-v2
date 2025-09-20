import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Lav rebaset vinduer [-50,+50] handelsdage for hver ref-dag")
    p.add_argument("--input", "-i", required=False, default=None, help="Input CSV eller Parquet (fil eller mappe) med features")
    p.add_argument("--out", "-o", required=False, default=None, help="Output mappe")
    p.add_argument("--per-ticker", action="store_true",
                   help="Gem én fil per ticker i stedet for samlet fil")
    p.add_argument("--before", type=int, default=50, help="Antal dage før (default=50)")
    p.add_argument("--after", type=int, default=50, help="Antal dage efter (default=50)")
    p.add_argument("--float-dp", type=int, default=None, help="Antal decimaler i CSV (Parquet bevarer fuld præcision)")
    p.add_argument("--format", choices=["csv","parquet","both"], default="csv", help="Output format")
    p.add_argument("--preset", choices=["standard"], default=None, help="Forudindstillet kørsel: standard")
    return p.parse_args()

def _flag_provided(*names: str) -> bool:
    import sys as _sys
    return any(n in _sys.argv for n in names)

def apply_preset(args):
    if not args.preset:
        return args
    if args.preset == "standard":
        # standard forventer compute_features preset-output
        if args.input is None and not _flag_provided("--input","-i"):
            # partitioneret mappe fra compute_features standard
            default_dir = "features.parquet"
            default_file = "features.parquet"  # hvis nogen bruger fil, lad det være samme
            if os.path.isdir(default_dir):
                args.input = default_dir
            elif os.path.isfile(default_file):
                args.input = default_file
            else:
                args.input = default_dir
        if args.out is None and not _flag_provided("--out","-o"):
            args.out = "rebased"
        if not _flag_provided("--before"):
            args.before = 75
        if not _flag_provided("--after"):
            args.after = 25
        if not _flag_provided("--per-ticker"):
            args.per_ticker = True
        if args.float_dp is None and not _flag_provided("--float-dp"):
            args.float_dp = 4
        if not _flag_provided("--format"):
            args.format = "csv"
    return args

def load_data(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Ticker","Date"]).drop_duplicates(subset=["Ticker","Date"], keep="last").reset_index(drop=True)
    # Sikr numeriske kolonner for værdi-beregning
    value_cols = [c for c in ["AdjClose","Close","MA_20","MA_50","MA_200"] if c in df.columns]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_windows_for_ticker(g: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
    value_cols = [c for c in ["AdjClose","Close","MA_20","MA_50","MA_200"] if c in g.columns]
    rsi_cols = [c for c in g.columns if c.upper().startswith("RSI")]

    rows_out = []
    n = len(g)
    for center in range(before, n - after):
        ref_row = g.iloc[center]
        ref_date = ref_row["Date"]
        ref_vals = {c: ref_row[c] for c in value_cols}
        # Skip hvis ref mangler værdier eller er 0/ikke-positiv (undgå deling med 0)
        if any(pd.isna(v) or (isinstance(v, (int,float,np.floating)) and v == 0) for v in ref_vals.values()):
            continue

        window = g.iloc[center-before:center+after+1].copy()
        offsets = np.arange(-before, after+1, dtype=int)

        out = pd.DataFrame({
            "Ticker": g["Ticker"].iloc[0],
            "RefDate": ref_date,
            "Offset": offsets,
            "Date": window["Date"].to_numpy(),
        })
        for c in value_cols:
            out[f"{c}_Rebased"] = (window[c].to_numpy() / ref_vals[c]) * 100.0
        for c in rsi_cols:
            out[c] = window[c].to_numpy()
        # Drop rækker hvor alle rebased value-kolonner er NaN (ingen pris for den offset-dato)
        rebased_cols = [f"{c}_Rebased" for c in value_cols]
        mask_keep = ~(out[rebased_cols].isna().all(axis=1)) if rebased_cols else pd.Series(True, index=out.index)
        out = out.loc[mask_keep].reset_index(drop=True)
        if not out.empty:
            rows_out.append(out)

    return pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Fast orden: Ticker, RefDate, Offset, Date, *_Rebased (AdjClose, Close, MA_20, MA_50, MA_200), RSI_*
    base = ["Ticker","RefDate","Offset","Date"]
    rebased_order = [
        "AdjClose_Rebased","Close_Rebased","MA_20_Rebased","MA_50_Rebased","MA_200_Rebased"
    ]
    rebased_cols = [c for c in rebased_order if c in df.columns]
    # RSI-sortering numerisk
    rsi_cols = [c for c in df.columns if c.upper().startswith("RSI")]
    def _num_sfx(name: str, prefix: str) -> float:
        try:
            return float(name[len(prefix):])
        except Exception:
            return float('inf')
    rsi_sorted = sorted(rsi_cols, key=lambda c: _num_sfx(c.upper(), "RSI_"))
    rest = [c for c in df.columns if c not in base and c not in rebased_cols and c not in rsi_sorted]
    cols = [c for c in base if c in df.columns] + rebased_cols + rsi_sorted + rest
    return df[cols]

def main():
    args = parse_args()
    # If run without any CLI args (e.g., VS Code "Run Python File"), default to preset=standard
    import sys as _sys
    if args.preset is None and len(_sys.argv) == 1:
        args.preset = "standard"
    args = apply_preset(args)
    if not args.input:
        raise SystemExit("Fejl: --input mangler. Angiv --input eller brug --preset standard.")
    if not args.out:
        raise SystemExit("Fejl: --out mangler. Angiv --out eller brug --preset standard.")
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.input)
    all_out = []

    for ticker, g in df.groupby("Ticker", sort=False, observed=False):
        print(f"Processing {ticker} ({len(g)} rows)...")
        rebased = make_windows_for_ticker(g, args.before, args.after)
        if rebased.empty:
            continue
        rebased = order_columns(rebased)
        if args.per_ticker:
            if args.format in ("csv","both"):
                out_file = Path(args.out) / f"{ticker}_rebased.csv"
                csv_float_format = None
                df_csv = rebased
                if args.float_dp is not None:
                    csv_float_format = f"%.{args.float_dp}f"
                    num_cols = [c for c in df_csv.columns if pd.api.types.is_numeric_dtype(df_csv[c])]
                    if num_cols:
                        df_csv = df_csv.copy()
                        df_csv[num_cols] = df_csv[num_cols].apply(pd.to_numeric, errors="coerce").round(args.float_dp)
                df_csv.to_csv(out_file, index=False, float_format=csv_float_format)
                print(f"  -> saved CSV {len(df_csv)} rows to {out_file}")
            if args.format in ("parquet","both"):
                out_pq = Path(args.out) / f"{ticker}_rebased.parquet"
                rebased.to_parquet(out_pq, index=False, compression="snappy")
                print(f"  -> saved Parquet {len(rebased)} rows to {out_pq}")
        else:
            all_out.append(rebased)

    if not args.per_ticker and all_out:
        combined = order_columns(pd.concat(all_out, ignore_index=True))
        if args.format in ("csv","both"):
            out_file = Path(args.out) / "rebased_all.csv"
            csv_float_format = None
            df_csv = combined
            if args.float_dp is not None:
                csv_float_format = f"%.{args.float_dp}f"
                num_cols = [c for c in df_csv.columns if pd.api.types.is_numeric_dtype(df_csv[c])]
                if num_cols:
                    df_csv = df_csv.copy()
                    df_csv[num_cols] = df_csv[num_cols].apply(pd.to_numeric, errors="coerce").round(args.float_dp)
            df_csv.to_csv(out_file, index=False, float_format=csv_float_format)
            print(f"Saved combined CSV with {len(df_csv)} rows to {out_file}")
        if args.format in ("parquet","both"):
            out_pq = Path(args.out) / "rebased_all.parquet"
            combined.to_parquet(out_pq, index=False, compression="snappy")
            print(f"Saved combined Parquet with {len(combined)} rows to {out_pq}")

if __name__ == "__main__":
    main()
