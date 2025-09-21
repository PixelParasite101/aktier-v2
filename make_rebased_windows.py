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
    p.add_argument(
        "--drop-rows-if-nan",
        choices=["none", "any", "all"],
        default="all",
        help=(
            "Styr rækker med NaN i rebased-kolonner: "
            "none=behold alle rækker; any=drop rækker hvor nogen rebased-kolonne er NaN; "
            "all=drop kun rækker hvor alle rebased-kolonner er NaN (default)."
        ),
    )
    p.add_argument(
        "--csv-head",
        type=int,
        default=None,
        help="Hvis sat: skriv kun de første N rækker til CSV (Parquet er altid komplet)",
    )
    p.add_argument(
        "--csv-tail",
        type=int,
        default=None,
        help="Hvis sat: skriv også de sidste N rækker til CSV (kombineret med --csv-head)",
    )
    p.add_argument(
        "--require-full-window",
        action="store_true",
        help=(
            "Kræv at hele vinduet [-before,+after] er komplet efter NaN-filtrering. "
            "Hvis ikke, springes vinduet/ref-dagen over. Giver første komplette datasæt fra offset -before."
        ),
    )
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
            args.format = "both"
        if not _flag_provided("--drop-rows-if-nan"):
            args.drop_rows_if_nan = "any"
        if not _flag_provided("--require-full-window"):
            args.require_full_window = True
        if args.csv_head is None and not _flag_provided("--csv-head"):
            args.csv_head = 1000
        if args.csv_tail is None and not _flag_provided("--csv-tail"):
            args.csv_tail = 1000
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
    # Adj/Close + alle MA_* der findes i input (dynamisk)
    ma_cols = [c for c in df.columns if c.startswith("MA_")]
    value_cols = [c for c in ["AdjClose", "Close"] if c in df.columns] + ma_cols
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_windows_for_ticker(
    g: pd.DataFrame,
    before: int,
    after: int,
    drop_policy: str = "all",
    require_full_window: bool = False,
) -> pd.DataFrame:
    # Rebase AdjClose/Close samt alle tilgængelige MA_* kolonner
    ma_cols = [c for c in g.columns if c.startswith("MA_")]
    value_cols = [c for c in ["AdjClose", "Close"] if c in g.columns] + ma_cols
    rsi_cols = [c for c in g.columns if c.upper().startswith("RSI")]

    rows_out = []
    n = len(g)
    for center in range(before, n - after):
        ref_row = g.iloc[center]
        ref_date = ref_row["Date"]
        ref_vals = {c: ref_row.get(c, pd.NA) for c in value_cols}
        # Vurder gyldige reference-værdier per kolonne (ikke-NaN og != 0)
        def _is_valid_ref(v):
            if pd.isna(v):
                return False
            try:
                return bool(v != 0)
            except Exception:
                return False
        valid_ref_cols = [c for c, v in ref_vals.items() if _is_valid_ref(v)]
        # Skip kun hvis INGEN reference-værdier er gyldige
        if not valid_ref_cols:
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
            col_name = f"{c}_Rebased"
            if c in valid_ref_cols:
                out[col_name] = (window[c].to_numpy() / ref_vals[c]) * 100.0
            else:
                # reference er NaN/0 — udfyld kolonne med NaN'er
                out[col_name] = np.nan
        for c in rsi_cols:
            out[c] = window[c].to_numpy()
        # Filtrer rækker baseret på NaN-politik for rebased-kolonner
        rebased_cols = [f"{c}_Rebased" for c in value_cols]
        if not rebased_cols:
            mask_keep = pd.Series(True, index=out.index)
        else:
            if drop_policy == "none":
                mask_keep = pd.Series(True, index=out.index)
            elif drop_policy == "any":
                # drop rækker hvor mindst én rebased-kolonne er NaN
                mask_keep = ~out[rebased_cols].isna().any(axis=1)
            else:  # "all" (default)
                # drop rækker hvor alle rebased-kolonner er NaN
                mask_keep = ~out[rebased_cols].isna().all(axis=1)
        # Hvis vi kræver et komplet vindue, og masken ville droppe rækker, så skip dette vindue
        if require_full_window and not mask_keep.all():
            continue
        out = out.loc[mask_keep].reset_index(drop=True)
        if not out.empty:
            rows_out.append(out)

    return pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Fast orden: Ticker, RefDate, Offset, Date, *_Rebased (AdjClose, Close, MA_* sorteret numerisk), RSI_*
    base = ["Ticker","RefDate","Offset","Date"]
    rebased_fixed = ["AdjClose_Rebased", "Close_Rebased"]
    # Alle MA_*_Rebased, sorteret efter numerisk suffix
    def _num_sfx(name: str, prefix: str) -> float:
        try:
            return float(name[len(prefix):])
        except Exception:
            return float('inf')
    ma_rebased = [c for c in df.columns if c.startswith("MA_") and c.endswith("_Rebased")]
    ma_sorted = sorted(ma_rebased, key=lambda c: _num_sfx(c[:-9], "MA_"))  # strip _Rebased
    rebased_cols = [c for c in rebased_fixed if c in df.columns] + ma_sorted
    # RSI-sortering numerisk
    rsi_cols = [c for c in df.columns if c.upper().startswith("RSI")]
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
        rebased = make_windows_for_ticker(
            g,
            args.before,
            args.after,
            drop_policy=args.drop_rows_if_nan,
            require_full_window=args.require_full_window,
        )
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
                # CSV sampling: keep only head/tail if requested
                if args.csv_head is not None or args.csv_tail is not None:
                    h = max(0, args.csv_head or 0)
                    t = max(0, args.csv_tail or 0)
                    n = len(df_csv)
                    if h + t >= n:
                        df_sample = df_csv
                    else:
                        idx_keep = set(range(h)) | set(range(max(0, n - t), n))
                        df_sample = df_csv.iloc[sorted(idx_keep)]
                    df_csv = df_sample
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
            # CSV sampling for combined as well
            if args.csv_head is not None or args.csv_tail is not None:
                h = max(0, args.csv_head or 0)
                t = max(0, args.csv_tail or 0)
                n = len(df_csv)
                if h + t >= n:
                    df_sample = df_csv
                else:
                    idx_keep = set(range(h)) | set(range(max(0, n - t), n))
                    df_sample = df_csv.iloc[sorted(idx_keep)]
                df_csv = df_sample
            df_csv.to_csv(out_file, index=False, float_format=csv_float_format)
            print(f"Saved combined CSV with {len(df_csv)} rows to {out_file}")
        if args.format in ("parquet","both"):
            out_pq = Path(args.out) / "rebased_all.parquet"
            combined.to_parquet(out_pq, index=False, compression="snappy")
            print(f"Saved combined Parquet with {len(combined)} rows to {out_pq}")

if __name__ == "__main__":
    main()
