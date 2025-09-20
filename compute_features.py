# compute_features.py
# Læser rå OHLCV (Ticker, Date, Open, High, Low, Close, AdjClose, Volume)
# og beregner MA'er (default: 20,50,200) samt RSI (default: 14) pr. ticker.
# Gemmer som Parquet (og evt. CSV).

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Beregn MA og RSI på aktiedata pr. ticker.")
    p.add_argument("--input", "-i", required=False, default=None, help="Sti til input (CSV eller Parquet).")
    p.add_argument("--out", "-o", required=False, default=None, help="Sti til output Parquet-fil eller mappe (ved partitionering).")
    p.add_argument("--csv", help="(Valgfrit) Skriv også en CSV-kopi hertil.")
    p.add_argument("--ma", nargs="+", type=int, default=[20, 50, 200],
                   help="MA-vinduer (dage). Default: 20 50 200")
    p.add_argument("--rsi", type=int, default=14, help="RSI-længde. Default: 14")
    p.add_argument("--use-adjclose", action="store_true",
                   help="Beregn indikatorer på AdjClose i stedet for Close.")
    p.add_argument("--partition-by", default=None,
                   help="Parquet partitionering, fx 'Ticker'. Hvis sat, skriv til mappe.")
    p.add_argument(
        "--float-dp",
        type=int,
        default=None,
        help="Antal decimaler for floats i CSV (Parquet bevarer fuld præcision).",
    )
    p.add_argument(
        "--preset",
        choices=["standard"],
        default=None,
        help="Forudindstillet kørsel: standard. Manuelle flags kan stadig override.",
    )
    return p.parse_args()

def _flag_provided(*names: str) -> bool:
    import sys as _sys
    return any(n in _sys.argv for n in names)

def apply_preset(args):
    if not args.preset:
        return args
    if args.preset == "standard":
        if args.input is None and not _flag_provided("--input", "-i"):
            # Standard preset i fetch skriver partitioneret Parquet
            default_dir = "data_all/history_all_parquet"
            default_file = "data_all/history_all.parquet"
            if os.path.isdir(default_dir):
                args.input = default_dir
            elif os.path.isfile(default_file):
                args.input = default_file
            else:
                # brug dir som default og lad read_input fejle hvis den ikke findes
                args.input = default_dir
        if args.out is None and not _flag_provided("--out", "-o"):
            args.out = "features.parquet"
        if not _flag_provided("--partition-by"):
            args.partition_by = "Ticker"
        if not _flag_provided("--csv"):
            args.csv = "features.csv"
        if not _flag_provided("--use-adjclose"):
            args.use_adjclose = True
        if args.float_dp is None and not _flag_provided("--float-dp"):
            args.float_dp = 4
    return args

def read_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet" or os.path.isdir(path):
        # tillad også en parquet-mappe
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input skal være .csv, .parquet eller en parquet-mappe.")
    # Grundtjek
    need = {"Ticker","Date","Close","AdjClose"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i input: {sorted(missing)}")
    # Dato som datetime og sortering
    df["Date"] = pd.to_datetime(df["Date"], utc=False, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Ticker","Date"]).drop_duplicates(subset=["Ticker","Date"], keep="last").reset_index(drop=True)
    return df

def rsi_wilder(close: pd.Series, length: int) -> pd.Series:
    """RSI med Wilder smoothing (EMA med alpha=1/length)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing via ewm
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # håndter 0-tab (avg_loss=0) → RSI=100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi

def add_features(df: pd.DataFrame, ma_windows: List[int], rsi_len: int, use_adj: bool) -> pd.DataFrame:
    price_col = "AdjClose" if use_adj else "Close"
    # Drop rækker uden pris før beregning, så indikatorer kun beregnes på handelsdage
    df = df.dropna(subset=[price_col]).copy()
    parts = []
    for ticker, g in df.groupby("Ticker", sort=False, observed=False):
        g = g.copy()
        # MA'er
        for w in ma_windows:
            g[f"MA_{w}"] = g[price_col].rolling(window=w, min_periods=w).mean()
        # RSI
        g[f"RSI_{rsi_len}"] = rsi_wilder(g[price_col], rsi_len)
        # Maskér RSI hvor prisen er NaN (burde være sjældent efter dropna, men for sikkerhed)
        g.loc[g[price_col].isna(), f"RSI_{rsi_len}"] = pd.NA
        parts.append(g)
    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a consistent column order: Ticker, Date, OHLCV/AdjClose, then indicators and any others."""
    preferred = [
        "Ticker", "Date",
        "Open", "High", "Low", "Close", "AdjClose", "Volume",
    ]
    # Indicators first (deterministic order: MA_*, then RSI_*)
    def _numeric_suffix(c: str, prefix: str) -> float:
        try:
            return float(c[len(prefix):])
        except Exception:
            return float('inf')
    ma_cols = [c for c in df.columns if c.startswith("MA_")]
    rsi_cols = [c for c in df.columns if c.startswith("RSI_")]
    indicators = sorted(ma_cols, key=lambda c: _numeric_suffix(c, "MA_")) + \
                 sorted(rsi_cols, key=lambda c: _numeric_suffix(c, "RSI_"))
    rest = [c for c in df.columns if c not in preferred and c not in indicators]
    ordered = [c for c in preferred if c in df.columns] + indicators + rest
    return df[ordered]

def write_parquet(df: pd.DataFrame, out_path: str, partition_by: Optional[str]):
    if partition_by:
        # skriv til mappe ved partitionering
        os.makedirs(out_path, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="snappy", partition_cols=[partition_by])
    else:
        # skriv enkeltfil
        df.to_parquet(out_path, index=False, compression="snappy")

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

    # Læs
    df = read_input(args.input)

    # Vælg kolonne til indikatorer
    price_base = "AdjClose" if args.use_adjclose else "Close"
    if not np.issubdtype(df[price_base].dtype, np.number):
        df[price_base] = pd.to_numeric(df[price_base], errors="coerce")

    # Beregn features
    feats = add_features(df, args.ma, args.rsi, use_adj=args.use_adjclose)
    # Fast kolonnerækkefølge (Ticker, Date, OHLCV/AdjClose, indikatorer)
    feats = order_columns(feats)

    # Skriv Parquet
    write_parquet(feats, args.out, args.partition_by)

    # (Valgfrit) CSV
    if args.csv:
        # Hvis out er en mappe (partitioneret), giver CSV mening som én samlet fil
        # Bemærk: kan fylde meget for store datasæt
        csv_float_format = None
        feats_csv = feats
        if args.float_dp is not None:
            csv_float_format = f"%.{args.float_dp}f"
            # afrund numeriske kolonner (inkl. inputpriser og beregnede indikatorer)
            numeric_cols = [
                c for c in feats.columns
                if pd.api.types.is_numeric_dtype(feats[c])
            ]
            if numeric_cols:
                feats_csv = feats.copy()
                feats_csv[numeric_cols] = feats_csv[numeric_cols].apply(pd.to_numeric, errors="coerce").round(args.float_dp)
        # Sikr samme kolonneorden i CSV
        feats_csv = order_columns(feats_csv)
        feats_csv.to_csv(args.csv, index=False, float_format=csv_float_format)

    # Lille konsol-opsummering
    cols_added = [c for c in feats.columns if c.startswith("MA_") or c.startswith("RSI_")]
    tickers_n = feats["Ticker"].nunique()
    print(f"OK. Tickers: {tickers_n}  Nye kolonner: {cols_added}")

if __name__ == "__main__":
    main()
