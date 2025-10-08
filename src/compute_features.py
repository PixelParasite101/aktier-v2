"""
compute_features implementation in src package.

Top-level `compute_features.py` will be a thin wrapper delegating to this module.
"""
# If executed directly from an editor (VS Code Run), ensure project root is on sys.path
# so imports like `from utils.common` resolve. When run as a module (python -m src.compute_features)
# this is a no-op.
import os
import sys
proj_root = os.path.dirname(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from utils.common import round_for_csv, write_metadata
import logging

logger = logging.getLogger("aktier.features")

def parse_args():
    p = argparse.ArgumentParser(description="Beregn MA og RSI på aktiedata pr. ticker.")
    p.add_argument("--input", "-i", required=False, default=None, help="Sti til input (CSV eller Parquet).")
    p.add_argument("--out", "-o", required=False, default=None, help="Sti til output Parquet-fil eller mappe (ved partitionering).")
    p.add_argument("--csv", help="(Valgfrit) Skriv også en CSV-kopi hertil.")
    p.add_argument("--ma", nargs="+", type=int, default=[20, 50, 200], help="MA-vinduer (dage). Default: 20 50 200")
    p.add_argument("--rsi", type=int, default=14, help="RSI-længde. Default: 14")
    p.add_argument("--use-adjclose", action="store_true", help="Beregn indikatorer på AdjClose i stedet for Close.")
    p.add_argument("--partition-by", default=None, help="Parquet partitionering, fx 'Ticker'. Hvis sat, skriv til mappe.")
    p.add_argument("--float-dp", type=int, default=None, help="Antal decimaler for floats i CSV (Parquet bevarer fuld præcision).")
    p.add_argument("--preset", choices=["standard"], default=None, help="Forudindstillet kørsel: standard. Manuelle flags kan stadig override.")
    p.add_argument("--show-config", action="store_true", help="Print den endelige konfiguration (efter preset) som JSON og exit.")
    p.add_argument("--config-out", default=None, help="Hvis sat: skriv den resolved konfiguration som JSON til denne fil og exit.")
    p.add_argument("--log-file", default=None, help="JSON-lines logfil (append).")
    return p.parse_args()

def _flag_provided(*names: str) -> bool:
    import sys as _sys
    return any(n in _sys.argv for n in names)

def apply_preset(args):
    if not args.preset:
        return args
    if args.preset == "standard":
        if args.input is None and not _flag_provided("--input", "-i"):
            # Default fetch output lives under data/fetch_data_all
            default_dir = os.path.join("data", "fetch_data_all", "history_all_parquet")
            default_file = os.path.join("data", "fetch_data_all", "history_all.parquet")
            if os.path.isdir(default_dir):
                args.input = default_dir
            elif os.path.isfile(default_file):
                args.input = default_file
            else:
                args.input = default_dir
        if args.out is None and not _flag_provided("--out", "-o"):
            # Write default parquet into its own folder under data (data/features)
            args.out = os.path.join("data", "features", "features.parquet")
        if not _flag_provided("--partition-by"):
            args.partition_by = "Ticker"
        if not _flag_provided("--csv"):
            # CSV preset under data/features as well
            args.csv = os.path.join("data", "features", "features.csv")
        if not _flag_provided("--use-adjclose"):
            args.use_adjclose = True
        if args.float_dp is None and not _flag_provided("--float-dp"):
            args.float_dp = 4
    return args

def read_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet" or os.path.isdir(path):
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input skal være .csv, .parquet eller en parquet-mappe.")
    need = {"Ticker","Date","Close","AdjClose"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i input: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], utc=False, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Ticker","Date"]).drop_duplicates(subset=["Ticker","Date"], keep="last").reset_index(drop=True)
    return df

def rsi_wilder(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi

def add_features(df: pd.DataFrame, ma_windows: List[int], rsi_len: int, use_adj: bool) -> pd.DataFrame:
    price_col = "AdjClose" if use_adj else "Close"
    df = df.dropna(subset=[price_col]).copy()
    parts = []
    for ticker, g in df.groupby("Ticker", sort=False, observed=False):
        g = g.copy()
        for w in ma_windows:
            g[f"MA_{w}"] = g[price_col].rolling(window=w, min_periods=w).mean()
        g[f"RSI_{rsi_len}"] = rsi_wilder(g[price_col], rsi_len)
        g.loc[g[price_col].isna(), f"RSI_{rsi_len}"] = pd.NA
        parts.append(g)
    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
    def _numeric_suffix(c: str, prefix: str) -> float:
        try:
            return float(c[len(prefix):])
        except Exception:
            return float('inf')
    ma_cols = [c for c in df.columns if c.startswith("MA_")]

    rsi_cols = [c for c in df.columns if c.startswith("RSI_")]
    indicators = sorted(ma_cols, key=lambda c: _numeric_suffix(c, "MA_")) + sorted(rsi_cols, key=lambda c: _numeric_suffix(c, "RSI_"))
    rest = [c for c in df.columns if c not in preferred and c not in indicators]
    ordered = [c for c in preferred if c in df.columns] + indicators + rest
    return df[ordered]

def write_parquet(df: pd.DataFrame, out_path: str, partition_by: Optional[str]):
    if partition_by:
        os.makedirs(out_path, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="snappy", partition_cols=[partition_by])
    else:
        df.to_parquet(out_path, index=False, compression="snappy")

def main():
    args = parse_args()
    import sys as _sys
    if args.preset is None and len(_sys.argv) == 1:
        args.preset = "standard"
    args = apply_preset(args)
    try:
        from utils.log import init_logging
        init_logging(getattr(args, "log_file", None))
    except Exception:
        pass
    if getattr(args, "show_config", False):
        import json as _json
        cfg = _json.dumps(vars(args), default=str, sort_keys=True, ensure_ascii=False, indent=2)
        print(cfg)
        if args.config_out:
            with open(args.config_out, "w", encoding="utf-8") as _f:
                _f.write(cfg)
        return
    if not args.input:
        raise SystemExit("Fejl: --input mangler. Angiv --input eller brug --preset standard.")
    if not args.out:
        raise SystemExit("Fejl: --out mangler. Angiv --out eller brug --preset standard.")
    df = read_input(args.input)
    price_base = "AdjClose" if args.use_adjclose else "Close"
    if not np.issubdtype(df[price_base].dtype, np.number):
        df[price_base] = pd.to_numeric(df[price_base], errors="coerce")
    feats = add_features(df, args.ma, args.rsi, use_adj=args.use_adjclose)
    feats = order_columns(feats)
    write_parquet(feats, args.out, args.partition_by)
    if args.csv:
        numeric_cols = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
        feats_csv, csv_float_format = round_for_csv(feats, args.float_dp, include_cols=numeric_cols)
        feats_csv = order_columns(feats_csv)
        feats_csv.to_csv(args.csv, index=False, float_format=csv_float_format)
    cols_added = [c for c in feats.columns if c.startswith("MA_") or c.startswith("RSI_")]
    tickers_n = feats["Ticker"].nunique()
    try:
        write_metadata(os.path.dirname(args.out) if not os.path.isdir(args.out) else args.out, name="features", args=args, extra={"tickers": tickers_n, "indicators": cols_added})
    except Exception:
        pass
    print(f"OK. Tickers: {tickers_n}  Nye kolonner: {cols_added}")

if __name__ == "__main__":
    main()
