#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analog_matcher_watch.py

Workflow:
  - Læs watch.csv (kolonne: Ticker)
  - For hver ticker:
      * Hent 200+lookback handelsdage (yfinance)
      * Beregn MA_20/50/200
      * Rebase valgte kolonner mod dag 0 (seneste dag)
      * Sammenlign lookback-vindue med alle reference-vinduer i rebased/ (per-ticker parquet)
      * Returnér topN matches og deres future (+1..horizon)
  - Gem per-query resultater (scores + futures) og evt. et samlet resume

Krav:
  pip install yfinance pandas numpy pyarrow matplotlib
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import logging
import sys
import threading
import time
from dataclasses import dataclass
from utils.common import write_metadata


# ----------------------------- Utils -----------------------------

def _ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Ukendt filtype: {path}")


def _parse_cols_list(s: str) -> List[str]:
    # "AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Tom --match-cols")
    return parts


def _parse_weights_list(s: str, n: int) -> List[float]:
    if not s:
        return [1.0] * n
    parts = [p.strip() for p in s.split(",") if p.strip()]
    w = [float(x) for x in parts]
    if len(w) != n:
        raise ValueError(f"Antal weights ({len(w)}) matcher ikke antal kolonner ({n}).")
    return w


def _safe_filename(name: str) -> str:
    import re
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]+', '_', name)
    name = name.strip().replace(' ', '_')
    return name or "NA"


def _score_pair(a: np.ndarray, b: np.ndarray, metric: str = "mse") -> float:
    if metric == "mse":
        return float(np.mean((a - b) ** 2))
    elif metric == "corr":
        if np.std(a) == 0 or np.std(b) == 0:
            return 1.0  # dårlig (ingen variation)
        corr = float(np.corrcoef(a, b)[0, 1])
        return 1.0 - corr  # lavere = bedre
    else:
        raise ValueError("Ukendt metric. Vælg 'mse' eller 'corr'.")


# ------------------------ Fetch & Features ------------------------

def fetch_stock_history_yf(ticker: str, min_trading_days: int = 300, interval: str = "1d") -> pd.DataFrame:
    """Hent ca. min_trading_days handelsdage (kalenderbuffer bruges for at sikre handelsdage)."""
    import yfinance as yf
    calendar_days = int(min_trading_days * 2.5)
    df = yf.download(ticker, period=f"{calendar_days}d", interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Ingen data hentet for {ticker}")
    df = df.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Adj Close": "AdjClose", "Volume": "Volume"
    })
    df = df.reset_index().rename(columns={"Date": "Date"})
    df = _ensure_datetime(df, "Date").sort_values("Date")
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in df.columns]
    df = df[cols]
    # Deduplicate potential duplicate columns from data source quirks
    if getattr(df.columns, "duplicated", None) is not None and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df.reset_index(drop=True)


def compute_mas_inplace(df: pd.DataFrame) -> None:
    if "AdjClose" not in df.columns:
        raise ValueError("AdjClose mangler i hentede data.")
    # Handle rare case where selecting 'AdjClose' yields a DataFrame due to non-unique columns
    s = df["AdjClose"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    df["MA_20"] = s.rolling(20).mean()
    df["MA_50"] = s.rolling(50).mean()
    df["MA_200"] = s.rolling(200).mean()


def _base_from_rebased_name(name: str) -> str:
    # "AdjClose_Rebased" -> "AdjClose", "MA_20_Rebased" -> "MA_20"
    if name.endswith("_Rebased"):
        return name[:-8]
    return name


def rebase_columns_inplace(df: pd.DataFrame, cols_rebased: Iterable[str]) -> None:
    """For hver col i cols_rebased (navne slutter på _Rebased), lav df[col] = (base / base[-1]) * 100."""
    df_num = df.copy()
    for col_rebased in cols_rebased:
        base = _base_from_rebased_name(col_rebased)
        if base not in df.columns:
            raise ValueError(f"Mangler basekolonne '{base}' for '{col_rebased}'.")
        s_base = df[base]
        # Handle rare non-unique column names returning a DataFrame
        if isinstance(s_base, pd.DataFrame):
            s_base = s_base.iloc[:, 0]
        s = pd.to_numeric(s_base, errors="coerce")
        ref = s.iloc[-1]
        if not np.isfinite(ref) or ref == 0:
            raise ValueError(f"Kan ikke rebase '{base}': sidste værdi er NaN/0.")
        df[col_rebased] = (s / ref) * 100.0


def build_query_vectors(df: pd.DataFrame, lookback: int, match_cols: List[str]) -> Dict[str, np.ndarray]:
    """Returnér en dict: kolonnenavn -> numpy array (length=lookback), uden NaN."""
    if len(df) < lookback:
        raise ValueError(f"Kun {len(df)} rækker hentet; kræver lookback={lookback}.")
    tail = df.tail(lookback).reset_index(drop=True)
    vecs: Dict[str, np.ndarray] = {}
    for col in match_cols:
        if col not in tail.columns:
            raise ValueError(f"Query mangler kolonnen '{col}'.")
        v = pd.to_numeric(tail[col], errors="coerce").to_numpy()
        if np.isnan(v).any():
            raise ValueError(f"Query kolonnen '{col}' indeholder NaN i de sidste {lookback} dage.")
        vecs[col] = v
    return vecs


# ----------------------- Reference scanning -----------------------

def iter_reference_windows(ref_dir: str,
                           match_cols: List[str],
                           lookback: int,
                           horizon: int):
    """
    Yield pr. vindue:
      (file_name, ticker, refdate, window_series_map {col->np.ndarray len=lookback}, future_map {col->np.ndarray len<=horizon})
    Kræver at reference-parquet har: Ticker, RefDate, Offset, Date, samt match_cols
    """
    d = Path(ref_dir)
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Ingen .parquet filer i {ref_dir}")

    lb_start, lb_end = -lookback + 1, 0  # eks.: -74..0 for lookback=75

    for fp in files:
        needed = {"Ticker", "RefDate", "Offset"} | set(match_cols)
        # Læs kun nødvendige kolonner (Parquet kolonneprojektion)
        try:
            df = pd.read_parquet(fp, columns=list(needed))
        except Exception:
            # Fallback: læs hele filen og subsettér
            df = pd.read_parquet(fp)
            df = df[[c for c in df.columns if c in needed]]
        if not needed.issubset(df.columns):
            # Spring filer over, der ikke matcher schemaet
            continue

        # Vinduesdel: -L+1..0
        df_lb = df[(df["Offset"] >= lb_start) & (df["Offset"] <= lb_end)]
        if df_lb.empty:
            continue

        for (ticker, refdate), g in df_lb.groupby(["Ticker", "RefDate"], sort=False):
            g = g.sort_values("Offset")
            ok = True
            win_map: Dict[str, np.ndarray] = {}
            for col in match_cols:
                s = pd.to_numeric(g[col], errors="coerce").to_numpy()
                if len(s) != lookback or np.isnan(s).any():
                    ok = False
                    break
                win_map[col] = s
            if not ok:
                continue

            # Fremtidsdel: 1..horizon (kan være kortere end horizon)
            df_full = df[(df["Ticker"] == ticker) & (df["RefDate"] == refdate)].sort_values("Offset")
            fut_map: Dict[str, np.ndarray] = {}
            for col in match_cols:
                fut = df_full[(df_full["Offset"] >= 1) & (df_full["Offset"] <= horizon)]
                arr = pd.to_numeric(fut[col], errors="coerce").to_numpy()
                # NaN i future er ok; vi tager dem med som NaN (kan filtreres senere hvis ønsket)
                fut_map[col] = arr
            yield (fp.name, ticker, pd.to_datetime(refdate), win_map, fut_map)


# --------------------------- Matching -----------------------------

def score_window(query_vecs: Dict[str, np.ndarray],
                 win_map: Dict[str, np.ndarray],
                 metric: str,
                 weights: List[float]) -> float:
    cols = list(query_vecs.keys())
    assert len(cols) == len(weights)
    total_w = float(sum(weights)) or 1.0
    score_sum = 0.0
    for col, w in zip(cols, weights):
        a = query_vecs[col]
        b = win_map[col]
        score_sum += w * _score_pair(a, b, metric=metric)
    return score_sum / total_w


def find_best_matches_for_query(query_vecs: Dict[str, np.ndarray],
                                ref_dir: str,
                                lookback: int,
                                horizon: int,
                                match_cols: List[str],
                                metric: str,
                                topn: int):
    scores = []
    futures = {}  # (ticker, refdate) -> {col->np.ndarray (future)}

    for fname, ticker, refdate, win_map, fut_map in iter_reference_windows(ref_dir, match_cols, lookback, horizon):
        try:
            score = score_window(query_vecs, win_map, metric=metric, weights=[1.0] * len(match_cols))
        except Exception:
            continue
        scores.append((ticker, refdate, score, fname))
        futures[(ticker, pd.to_datetime(refdate))] = fut_map

    if not scores:
        raise RuntimeError("Fandt ingen gyldige referencevinduer—tjek lookback, match-kolonner og schemaet i rebased/.")

    scores.sort(key=lambda x: x[2])
    top = scores[:topn]
    df_scores = pd.DataFrame(top, columns=["MatchTicker", "RefDate", "Score", "SourceFile"]).sort_values("Score")

    # Pak futures i "langt" format for alle match_cols
    rows = []
    for m_tkr, rdt, _, _ in top:
        fmaps = futures.get((m_tkr, pd.to_datetime(rdt)), {})
        max_len = max((len(arr) for arr in fmaps.values()), default=0)
        for i in range(max_len):
            row = {"MatchTicker": m_tkr, "RefDate": pd.to_datetime(rdt), "FutureOffset": i + 1}
            for col in match_cols:
                arr = fmaps.get(col, np.array([]))
                row[col] = float(arr[i]) if i < len(arr) and np.isfinite(arr[i]) else np.nan
            rows.append(row)
    df_future = pd.DataFrame(rows).sort_values(["RefDate", "FutureOffset"]).reset_index(drop=True)

    return df_scores.reset_index(drop=True), df_future


# ----------------------------- Plot ------------------------------

def plot_matches(query_vecs: Dict[str, np.ndarray],
                 df_future: pd.DataFrame,
                 topk: int = 5,
                 plot_col: str = "AdjClose_Rebased"):
    import matplotlib.pyplot as plt

    if plot_col not in query_vecs:
        raise ValueError(f"'{plot_col}' findes ikke i query. Vælg blandt: {list(query_vecs.keys())}")

    lookback = len(next(iter(query_vecs.values())))
    x_query = np.arange(-lookback + 1, 1)

    plt.figure(figsize=(9, 5))
    # Plot query
    plt.plot(x_query, query_vecs[plot_col], label=f"Query ({plot_col})")

    shown = 0
    for (tkr, rdt), g in df_future.groupby(["MatchTicker", "RefDate"]):
        fx = g["FutureOffset"].to_numpy()
        fy = pd.to_numeric(g[plot_col], errors="coerce").to_numpy()
        plt.plot(fx, fy, alpha=0.6, label=f"{tkr} {str(pd.to_datetime(rdt).date())} (+)")
        shown += 1
        if shown >= topk:
            break

    plt.axvline(0, linestyle="--")
    plt.xlabel("Offset (handelsdage)")
    plt.ylabel(f"{plot_col} (ref=100 på dag 0)")
    plt.title(f"Query vs. historiske 'fremtider' ({plot_col})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------ CLI ------------------------------

def _flag_provided(*names: str) -> bool:
    import sys as _sys
    return any(n in _sys.argv for n in names)


def apply_preset(args):
    if not getattr(args, 'preset', None):
        return args
    if args.preset == 'standard':
        if args.watch is None and not _flag_provided('--watch'):
            args.watch = 'watch.csv'
        if args.ref_dir == 'rebased' and not _flag_provided('--ref-dir'):
            args.ref_dir = 'rebased'
        if args.out_dir == 'analog_out' and not _flag_provided('--out-dir'):
            args.out_dir = 'analog_out'
        if args.match_cols == 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased' and not _flag_provided('--match-cols'):
            args.match_cols = 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased'
        if args.lookback == 75 and not _flag_provided('--lookback'):
            args.lookback = 75
        if args.horizon == 25 and not _flag_provided('--horizon'):
            args.horizon = 25
        if args.topn == 10 and not _flag_provided('--topn'):
            args.topn = 10
        # Auto index usage hvis filen findes (men kun hvis user ikke eksplicit har sat --use-index/--no-use-index)
        if not _flag_provided('--use-index'):
            # Vi sætter ikke flag endnu; det håndteres efter argument parsing (fil-eksistens tjek)
            setattr(args, '_auto_use_index', True)
    return args


def main():
    ap = argparse.ArgumentParser(description="Watch-list analog matching mod rebased reference-datasæt")
    # Make --watch optional here; we'll auto-fill sensible defaults when run without args
    ap.add_argument("--watch", required=False, default=None, help="CSV med kolonnen 'Ticker'")
    ap.add_argument("--ref-dir", default="rebased", help="Mappe med *rebased* per-ticker .parquet")
    ap.add_argument("--lookback", type=int, default=75, help="Antal handelsdage i query (L)")
    ap.add_argument("--horizon", type=int, default=25, help="Fremtidsdage fra reference (+1..H)")
    ap.add_argument("--topn", type=int, default=10, help="Antal bedste matches pr. query-ticker")
    ap.add_argument("--metric", choices=["mse", "corr"], default="mse", help="Match-metrik")
    ap.add_argument("--match-cols", default="AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased",
                    help="Komma-sep. kolonner at matche på (skal findes i reference).")
    ap.add_argument("--weights", default="", help="Komma-sep. vægte til match-cols (default: alle 1.0)")
    ap.add_argument("--out-dir", default="analog_out", help="Outputmappe til scores/futures")
    ap.add_argument("--plot", action="store_true", help="Vis plot for hver query (plotter første col i --match-cols)")
    ap.add_argument("--query-from-features", action="store_true",
                    help="Brug eksisterende features (features.parquet/CSV) for query i stedet for at hente fra yfinance")
    ap.add_argument("--query-features-dir", default="features.parquet",
                    help="Sti til features-parquet mappe/fil eller features.csv, når --query-from-features er sat")
    # Prefilter / index (Step A flags) – integreres i senere refaktor
    ap.add_argument("--use-index", action="store_true", help="Brug signatur-index til prefilter af kandidater")
    ap.add_argument("--index-file", default="rebased_index.parquet", help="Parquet fil med signatur index")
    ap.add_argument("--sig-downsample", type=int, default=8, help="Signatur downsample størrelse (skal matche index)")
    ap.add_argument("--prefilter-topk", type=int, default=500, help="Antal kandidater efter prefilter før fuld scoring")
    ap.add_argument("--prefilter-metric", choices=["mse","corr"], default="mse", help="Distance-metrik for signatur (mse|corr)")
    ap.add_argument("--cache-ref", action="store_true", help="Cache indlæste rebased filer i hukommelsen mellem queries")
    ap.add_argument("--cache-ref-size", type=int, default=32, help="Maks antal rebased filer i cache (LRU)")
    ap.add_argument("--min-corr", type=float, default=None, help="Filtrer endelige matches: kræver corr >= denne værdi (kun metric=corr)")
    ap.add_argument("--score-workers", type=int, default=1, help="Antal worker tråde for batch scoring (vectoriseret)")
    import sys as _sys
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging (debug).")
    ap.add_argument("--log-file", default=None, help="Optional JSON log file (adds structured log lines).")
    ap.add_argument("--heartbeat-interval", type=float, default=10.0,
                    help="Seconds between heartbeat messages to indicate the watcher is alive (0 to disable).")
    ap.add_argument("--dry-run", action="store_true", help="Do not call network or write files; simulate steps for testing and heartbeat.")
    ap.add_argument("--fast-fullscan", action="store_true", help="Aktiver vectoriseret full-scan (hurtigere) når --use-index ikke er sat.")
    ap.add_argument('--preset', choices=['standard'], default=None, help='Preset konfiguration (pt. kun standard).')
    ap.add_argument('--show-config', action='store_true', help='Vis resolved config og exit.')
    ap.add_argument('--config-out', default=None, help='Skriv resolved config til fil og exit.')
    args = ap.parse_args()

    # Auto preset hvis ingen arguments
    if args.preset is None and len(_sys.argv) == 1:
        args.preset = 'standard'
    args = apply_preset(args)

    # Hvis auto-index foreslået af preset og fil eksisterer -> aktiver
    if getattr(args, '_auto_use_index', False) and not args.use_index:
        if os.path.exists(args.index_file):
            args.use_index = True
        # Fjern internt markeringsflag
        try:
            delattr(args, '_auto_use_index')
        except Exception:
            pass

    if getattr(args, 'show_config', False):
        cfg_json = json.dumps({k: v for k, v in vars(args).items() if not k.startswith('_')},
                               default=str, sort_keys=True, ensure_ascii=False, indent=2)
        print(cfg_json)
        if args.config_out:
            with open(args.config_out, 'w', encoding='utf-8') as f:
                f.write(cfg_json)
        return

    if not args.watch:
        ap.error("the following arguments are required: --watch (eller brug --preset standard)")

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Try to use utils.log if available for JSON logging; fall back to basicConfig
    try:
        from utils.log import init_logging
        init_logging(log_file=args.log_file, level=log_level)
        log = logging.getLogger("aktier.analog")
    except Exception:
        logging.basicConfig(stream=sys.stdout, level=log_level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        log = logging.getLogger("aktier.analog")

    # Simple status object updated as the script progresses; reported by heartbeat
    @dataclass
    class _Status:
        current_ticker: Optional[str] = None
        step: str = "idle"
        last_msg: Optional[str] = None
        processed: int = 0
        total: int = 0
        ewma_seconds: Optional[float] = None

    status = _Status()

    # Heartbeat thread prints/logs a short status line periodically so the user sees progress
    def _heartbeat_loop(interval: float, stop_event: threading.Event):
        while not stop_event.wait(interval):
            try:
                prog = f"{status.processed}/{status.total}" if status.total else "-/-"
                eta = "-"
                if status.ewma_seconds is not None and status.processed < status.total and status.total:
                    remaining = status.total - status.processed
                    est = status.ewma_seconds * remaining
                    # format as H:MM:SS or M:SS
                    m, s = divmod(int(est), 60)
                    h, m = divmod(m, 60)
                    eta = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                msg = f"heartbeat: step={status.step} ticker={status.current_ticker or '-'} progress={prog} eta={eta} msg={status.last_msg or '-'}"
                log.info(msg)
            except Exception:
                # Best-effort heartbeat; do not crash the main program
                pass

    hb_stop = threading.Event()
    hb_thread = None
    if args.heartbeat_interval and args.heartbeat_interval > 0:
        hb_thread = threading.Thread(target=_heartbeat_loop, args=(args.heartbeat_interval, hb_stop), daemon=True)
        hb_thread.start()

    ref_dir = args.ref_dir
    lookback = args.lookback
    horizon = args.horizon
    topn = args.topn
    metric = args.metric
    match_cols = _parse_cols_list(args.match_cols)
    weights = _parse_weights_list(args.weights, len(match_cols)) if args.weights else [1.0] * len(match_cols)
    file_cache: Dict[str, pd.DataFrame] = {}
    file_cache_order: List[str] = []  # LRU
    MAX_CACHE = max(1, args.cache_ref_size)

    def _get_rebased_file_cached(fname: str) -> Optional[pd.DataFrame]:
        """LRU cache fetch for a rebased parquet file (full load)."""
        path = Path(ref_dir) / fname
        key = str(path.resolve())
        if args.cache_ref and key in file_cache:
            # move to end (most recent)
            try:
                file_cache_order.remove(key)
            except ValueError:
                pass
            file_cache_order.append(key)
            return file_cache[key]
        try:
            df_local = pd.read_parquet(path)
        except Exception:
            return None
        if args.cache_ref:
            file_cache[key] = df_local
            file_cache_order.append(key)
            # evict
            while len(file_cache_order) > MAX_CACHE:
                oldest = file_cache_order.pop(0)
                file_cache.pop(oldest, None)
        return df_local

    # Læs watch.csv
    watch_df = pd.read_csv(args.watch)
    if "Ticker" not in watch_df.columns:
        raise SystemExit("watch.csv skal have kolonnen 'Ticker'.")
    tickers = [str(t).strip() for t in watch_df["Ticker"].dropna().tolist() if str(t).strip()]
    if not tickers:
        raise SystemExit("watch.csv indeholder ingen tickere.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting analog matcher: watch=%s  ref_dir=%s  lookback=%d  horizon=%d preset=%s use_index=%s", args.watch, ref_dir, lookback, horizon, args.preset, args.use_index)

    # Hjælper: indlæs features for en enkelt ticker og nødvendige base-kolonner
    def _load_query_from_features(features_path: str, ticker: str, base_cols: List[str]) -> Optional[pd.DataFrame]:
        p = Path(features_path)
        use_cols = ["Ticker", "Date"] + sorted(set(base_cols))
        if p.is_dir():
            # Prøv parti under Ticker=<ticker>
            part_dir = p / f"Ticker={ticker}"
            if not part_dir.exists():
                return None
            files = sorted(part_dir.glob("*.parquet"))
            if not files:
                return None
            dfs = []
            for fp in files:
                try:
                    dfp = pd.read_parquet(fp, columns=[c for c in use_cols if c != "Ticker"])  # Ticker er implicit
                except Exception:
                    dfp = pd.read_parquet(fp)
                    dfp = dfp[[c for c in dfp.columns if c in use_cols or c == "Date"]]
                # Sikr Ticker-kolonne findes
                if "Ticker" not in dfp.columns:
                    dfp["Ticker"] = ticker
                dfs.append(dfp)
            if not dfs:
                return None
            df = pd.concat(dfs, ignore_index=True)
            df = df[df["Ticker"] == ticker]
        else:
            # Fil: parquet eller csv
            if p.suffix.lower() == ".parquet":
                try:
                    df = pd.read_parquet(p, columns=use_cols)
                except Exception:
                    df = pd.read_parquet(p)
                    df = df[[c for c in df.columns if c in use_cols]]
            elif p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p, usecols=lambda c: c in use_cols)
                except Exception:
                    df = pd.read_csv(p)
                    df = df[[c for c in df.columns if c in use_cols]]
            else:
                return None
            df = df[df["Ticker"] == ticker]

        if df is None or df.empty:
            return None
        df = _ensure_datetime(df, "Date").dropna(subset=["Date"]).sort_values("Date")
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        # Numerisk koercion for base-kolonner
        for c in base_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # Behandl hver query-ticker
    all_scores = []
    status.total = len(tickers)
    for q_ticker in tickers:
        status.current_ticker = q_ticker
        status.step = "start"
        status.last_msg = "begin processing"
        log.info("Processing query ticker: %s", q_ticker)
        # Byg query-data fra features (hvis valgt), ellers yfinance
        ticker_start = time.perf_counter()
        df: Optional[pd.DataFrame] = None
        try:
            if args.query_from_features:
                base_cols = [ _base_from_rebased_name(c) for c in match_cols ]
                status.step = "load_features"
                status.last_msg = f"attempting to load features for {q_ticker}"
                df = _load_query_from_features(args.query_features_dir, q_ticker, base_cols)
                if df is None or df.empty:
                    log.warning("Kunne ikke indlæse features for %s fra %s; falder tilbage til yfinance", q_ticker, args.query_features_dir)
                    status.last_msg = "features not found; falling back to yfinance"

            if df is None:
                # Hent 200+lookback handelsdage (for at kunne beregne MA_200 korrekt)
                min_days = 200 + lookback
                status.step = "fetch_yf"
                status.last_msg = f"fetching history (min_days={min_days})"
                log.debug("Henter historik fra yfinance for %s (min_days=%d)", q_ticker, min_days)
                if args.dry_run:
                    # Simulate minimal dataframe with required columns and enough rows
                    dates = pd.date_range(end=pd.Timestamp.today(), periods=min_days)
                    df = pd.DataFrame({"Date": dates, "AdjClose": 100.0 + pd.Series(range(len(dates))),
                                       "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1000})
                    # Compute moving averages so rebase columns (e.g., MA_20) exist
                    try:
                        compute_mas_inplace(df)
                    except Exception:
                        # best-effort in dry-run
                        pass
                else:
                    df = fetch_stock_history_yf(q_ticker, min_trading_days=min_days)
                    status.step = "compute_ma"
                    status.last_msg = "computing moving averages"
                    compute_mas_inplace(df)

            # Rebase de kolonner vi matcher på (baseret på sidste dato i datasættet)
            status.step = "rebase"
            status.last_msg = "rebasing columns"
            rebase_columns_inplace(df, cols_rebased=match_cols)

            # Byg query-vektorer (kun sidste lookback dage)
            status.step = "build_query"
            status.last_msg = "building query vectors"
            query_vecs = build_query_vectors(df, lookback=lookback, match_cols=match_cols)
            log.debug("Query vektorer bygget for %s (lookback=%d); cols=%s", q_ticker, lookback, match_cols)

            # Prefilter kandidater via signatur index hvis aktiveret
            candidate_set = None
            if args.use_index:
                status.step = "prefilter"
                status.last_msg = "loading index"
                t_pref_start = time.perf_counter()
                try:
                    idx = pd.read_parquet(args.index_file)
                except FileNotFoundError:
                    log.warning("Index fil ikke fundet (%s); fortsætter uden prefilter", args.index_file)
                    idx = None
                except Exception as e:
                    log.warning("Fejl ved indlæsning af index (%s): %s; fallback til full scan", args.index_file, e)
                    idx = None
                if idx is not None:
                    sig_cols = []
                    use_corr = (metric == 'corr')
                    for mc in match_cols:
                        sc_base = f"{mc}_sig"
                        sc_corr = f"{mc}_sig_z"
                        chosen_col = sc_corr if use_corr and sc_corr in idx.columns else sc_base
                        if chosen_col in idx.columns:
                            sig_cols.append(chosen_col)
                        else:
                            log.warning("Mangler signatur kolonne i index: %s (deaktiverer prefilter)", chosen_col)
                            sig_cols = []
                            break
                    if sig_cols:
                        # Downsample query lokalt (samme metode som i index builder)
                        def _downsample(arr: np.ndarray, n: int):
                            if n <= 0:
                                return np.array([], dtype=float)
                            if arr.size == 0:
                                return np.full(n, np.nan)
                            idx_lin = np.linspace(0, max(0, arr.size - 1), num=n, dtype=int)
                            return arr[idx_lin].astype(float, copy=False)
                        q_sigs = []
                        for mc in match_cols:
                            q_sigs.append(_downsample(query_vecs[mc], args.sig_downsample))
                        try:
                            q_sigs = np.stack(q_sigs, axis=0)  # (C,D)
                        except Exception:
                            q_sigs = None
                        dist_list = []
                        keep_rows = []
                        file_rows = []  # (ticker, refdate, sourcefile)
                        if q_sigs is not None:
                            import json as _json
                            for _, row in idx.iterrows():
                                sig_stack = []
                                valid = True
                                for sc in sig_cols:
                                    val = row[sc]
                                    if isinstance(val, str):
                                        try:
                                            val = _json.loads(val)
                                        except Exception:
                                            val = []
                                    arr = np.array(val, dtype=float)
                                    sig_stack.append(arr)
                                if len(sig_stack) != len(match_cols):
                                    continue
                                try:
                                    stack = np.stack(sig_stack, axis=0)
                                except Exception:
                                    continue
                                if stack.shape[1] != q_sigs.shape[1]:
                                    valid = False
                                if not valid:
                                    continue
                                if metric == 'mse':
                                    diff = stack - q_sigs
                                    dist_val = np.nanmean(diff * diff)
                                else:  # corr: hvis sig_cols er z-normaliserede er mean=0,std=1
                                    # Approksimer distance som gennemsnitlig (1 - (stack * q).mean over D pr. kolonne)
                                    prod = stack * q_sigs  # (C,D)
                                    corr_sig = np.nanmean(prod, axis=1)  # pr kolonne
                                    corr_sig = np.where(np.isnan(corr_sig), 0.0, corr_sig)
                                    dist_val = float(np.mean(1.0 - corr_sig))
                                if not np.isfinite(dist_val):
                                    continue
                                dist_list.append(dist_val)
                                keep_rows.append((row['Ticker'], pd.to_datetime(row['RefDate']), row['SourceFile']))
                                file_rows.append(row['SourceFile'])
                        if dist_list:
                            order = np.argsort(dist_list)
                            k = min(args.prefilter_topk, len(order))
                            chosen = set()
                            chosen_files = set()
                            for j in order[:k]:
                                tkr, rdt, sf = keep_rows[j]
                                chosen.add((tkr, rdt))
                                chosen_files.add(sf)
                            candidate_set = chosen
                            candidate_files = chosen_files
                            t_pref = time.perf_counter() - t_pref_start
                            log.info("Prefilter: rows=%d chosen=%d time=%.3fs", len(dist_list), len(chosen), t_pref)
                        else:
                            log.warning("Prefilter: ingen kandidater; full scan")
                status.last_msg = "prefilter done" if candidate_set is not None else "prefilter skipped"

            # Find matches mod reference
            # (brug weights ved scoring)
            def weighted_score_window(qv, wm):
                assert len(match_cols) == len(weights)
                total_w = float(sum(weights)) or 1.0
                s = 0.0
                for col, w in zip(match_cols, weights):
                    s += w * _score_pair(qv[col], wm[col], metric=metric)
                return s / total_w

            scores = []
            futures = {}
            status.step = "matching"
            status.last_msg = "scanning reference windows"
            if candidate_set is not None:
                # Vectoriseret scoring: læs kun filer der indeholder kandidater
                # Gruppér kandidater efter SourceFile ved opslag i rebased fil (vi kender ikke SourceFile direkte -> læs alle og filtrér)
                # Simpel implementering: stadig brug iter_reference_windows men skipper non-kandidater, akkumuler win_map arrays og scorer batchvis pr. chunk
                # (Fremtidig optimering: bygge mapping Ticker,RefDate->SourceFile fra index og læse filer én gang.)
                batch_cols = match_cols
                for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(ref_dir, match_cols, lookback, horizon):
                    if 'candidate_files' in locals() and candidate_files and fname not in candidate_files:
                        continue
                    key_pair = (r_ticker, pd.to_datetime(refdate))
                    if key_pair not in candidate_set:
                        continue
                    # Saml arrays for vectoriseret scoring senere
                    try:
                        scores.append((q_ticker, r_ticker, refdate, win_map, fname))
                        futures[(q_ticker, r_ticker, pd.to_datetime(refdate))] = fut_map
                    except Exception:
                        continue
                # Udfør scoring i én NumPy operation pr. kolonne
                if scores:
                    # scores midlertidig struktur: (q_ticker, r_ticker, refdate, win_map, fname)
                    win_arrays = {col: [] for col in batch_cols}
                    meta = []
                    for _, r_ticker, refdate, win_map, fname in scores:
                        ok = True
                        for col in batch_cols:
                            arr = win_map[col]
                            if arr.shape[0] != lookback:
                                ok = False
                                break
                        if not ok:
                            continue
                        for col in batch_cols:
                            win_arrays[col].append(win_map[col])
                        meta.append((r_ticker, refdate, fname))
                    if meta:
                        # Stack arrays
                        col_scores = []
                        total_w = float(sum(weights)) or 1.0
                        N = len(meta)
                        workers = max(1, args.score_workers)
                        chunk_size = max(1, (N + workers - 1) // workers) if workers > 1 else N
                        from concurrent.futures import ThreadPoolExecutor

                        def _score_chunk(col, w, start, end):
                            A = np.stack(win_arrays[col][start:end], axis=0)
                            q = query_vecs[col][None, :]
                            if metric == 'mse':
                                diff = A - q
                                return w * np.mean(diff * diff, axis=1)
                            else:
                                a_center = A - A.mean(axis=1, keepdims=True)
                                q_center = q - q.mean(axis=1, keepdims=True)
                                denom = (np.sqrt((a_center*a_center).sum(axis=1)) * np.sqrt((q_center*q_center).sum(axis=1)))
                                denom = np.where(denom == 0, np.nan, denom)
                                num = (a_center * q_center).sum(axis=1)
                                corr = num / denom
                                corr = np.where(np.isnan(corr), 0.0, corr)
                                return w * (1.0 - corr)

                        for col, w in zip(batch_cols, weights):
                            if workers == 1 or N < 512:
                                # Single chunk
                                col_scores.append(_score_chunk(col, w, 0, N))
                            else:
                                results = []
                                with ThreadPoolExecutor(max_workers=workers) as ex:
                                    futures_exec = []
                                    for start in range(0, N, chunk_size):
                                        end = min(N, start + chunk_size)
                                        futures_exec.append(ex.submit(_score_chunk, col, w, start, end))
                                    for fut in futures_exec:
                                        results.append(fut.result())
                                col_scores.append(np.concatenate(results, axis=0))
                        final_scores = sum(col_scores) / total_w
                        # Parallelisering (eksperimentel): chunk resultatberegning hvis workers>1 (landingsplads for fremtidig forbedring)
                        # Lige nu er beregning allerede udført; fremtidig version kan parallellisere per kolonne eller per chunk af A.
                        # Rekonstruer endelige scores-listen i original struktur
                        new_scores = []
                        for (r_ticker, refdate, fname), sc in zip(meta, final_scores):
                            new_scores.append((q_ticker, r_ticker, refdate, float(sc), fname))
                        scores = new_scores
                    else:
                        scores = []
                else:
                    scores = []
            else:
                # Full scan mode
                if args.fast_fullscan:
                    # Vectoriseret scoring per fil (læs lookback vinduer samlet og pivot/unstack)
                    lb_start, lb_end = -lookback + 1, 0
                    from collections import defaultdict
                    for fp in sorted(Path(ref_dir).glob("*.parquet")):
                        try:
                            needed = ["Ticker", "RefDate", "Offset"] + match_cols
                            dfp = pd.read_parquet(fp, columns=needed)
                        except Exception:
                            try:
                                dfp = pd.read_parquet(fp)
                                dfp = dfp[[c for c in dfp.columns if c in needed]]
                            except Exception:
                                continue
                        base_needed = {"Ticker", "RefDate", "Offset"}
                        if not base_needed.issubset(dfp.columns):
                            continue
                        df_look = dfp[(dfp["Offset"] >= lb_start) & (dfp["Offset"] <= lb_end)].copy()
                        if df_look.empty:
                            continue
                        # Pivot hver kolonne -> (windows x lookback)
                        # Saml index tiltænkt: MultiIndex (Ticker, RefDate)
                        pivot_mats = {}
                        valid_windows = None
                        offsets_expected = list(range(lb_start, lb_end + 1))
                        for col in match_cols:
                            if col not in df_look.columns:
                                pivot_mats = {}
                                break
                            sub = df_look[["Ticker", "RefDate", "Offset", col]].copy()
                            try:
                                mat = sub.pivot_table(index=["Ticker", "RefDate"], columns="Offset", values=col, dropna=False)
                            except Exception:
                                pivot_mats = {}
                                break
                            # Sikr alle offsets findes som kolonner
                            missing_off_cols = [o for o in offsets_expected if o not in mat.columns]
                            if missing_off_cols:
                                # Tilføj NaN kolonner for at kunne filtrere dem væk senere
                                for o in missing_off_cols:
                                    mat[o] = np.nan
                            # Reorder columns in chronological order
                            mat = mat[offsets_expected]
                            # Drop vinduer med NaN eller forkert længde
                            mat = mat.dropna(axis=0, how='any')
                            if mat.empty:
                                pivot_mats = {}
                                break
                            if valid_windows is None:
                                valid_windows = mat.index
                            else:
                                # Intersektion for at sikre fælles vinduer på tværs af kolonner
                                valid_windows = valid_windows.intersection(mat.index)
                            pivot_mats[col] = mat
                        if not pivot_mats:
                            continue
                        # Reducér til fælles vinduer
                        for col in list(pivot_mats.keys()):
                            pivot_mats[col] = pivot_mats[col].loc[valid_windows]
                            if pivot_mats[col].empty:
                                del pivot_mats[col]
                        if len(pivot_mats) != len(match_cols):
                            continue
                        nW = len(valid_windows)
                        if nW == 0:
                            continue
                        # Vectoriser scoringsberegning
                        per_col_scores = []
                        total_w = float(sum(weights)) or 1.0
                        for col, w in zip(match_cols, weights):
                            A = pivot_mats[col].to_numpy(dtype=float)  # shape (nW, lookback)
                            q = query_vecs[col][None, :]  # (1, L)
                            if metric == 'mse':
                                diff = A - q
                                col_score = np.mean(diff * diff, axis=1)
                            else:  # corr
                                A_center = A - A.mean(axis=1, keepdims=True)
                                q_center = q - q.mean(axis=1, keepdims=True)
                                denom = (np.sqrt((A_center*A_center).sum(axis=1)) * np.sqrt((q_center*q_center).sum(axis=1)))
                                denom = np.where(denom == 0, np.nan, denom)
                                num = (A_center @ q_center.T).ravel()
                                corr = num / denom
                                corr = np.where(np.isnan(corr), 0.0, corr)
                                col_score = 1.0 - corr
                            per_col_scores.append(w * col_score)
                        final_scores = sum(per_col_scores) / total_w
                        # Tilføj til samlet liste
                        for (tic, rdt), sc in zip(valid_windows, final_scores):
                            scores.append((q_ticker, tic, pd.to_datetime(rdt), float(sc), fp.name))
                else:
                    # Fallback: eksisterende iterative metode
                    for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(ref_dir, match_cols, lookback, horizon):
                        try:
                            s = weighted_score_window(query_vecs, win_map)
                        except Exception:
                            continue
                        scores.append((q_ticker, r_ticker, refdate, s, fname))
                        futures[(q_ticker, r_ticker, pd.to_datetime(refdate))] = fut_map

            if args.use_index:
                total_candidates = len(candidate_set) if candidate_set is not None else 0
                log.info("Scan summary: query=%s scored_windows=%d candidate_windows=%s prefilter=%s", \
                         q_ticker, len(scores), total_candidates if candidate_set is not None else '-', \
                         'on' if candidate_set is not None else 'off')

            # Ryd ikke cache mellem tickere; global reuse.

            if not scores:
                log.warning("Ingen gyldige referencevinduer fundet for %s", q_ticker)
                status.last_msg = "no valid reference windows"
                # continue inside try will jump to finally
                continue

            scores.sort(key=lambda x: x[3])  # lavest først (Score = distance eller 1-corr)
            top = scores[:topn]
            df_scores = pd.DataFrame(top, columns=["QueryTicker", "MatchTicker", "RefDate", "Score", "SourceFile"])

            # Hvis vi ikke allerede har futures (vectoriseret path) så hent kun for top vinduer
            if args.use_index is False and args.fast_fullscan and not futures:
                wanted = {(m, pd.to_datetime(rdt)) for _, m, rdt, _, _ in top}
                for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(ref_dir, match_cols, lookback, horizon):
                    key = (r_ticker, pd.to_datetime(refdate))
                    if key in wanted and (q_ticker, r_ticker, pd.to_datetime(refdate)) not in futures:
                        futures[(q_ticker, r_ticker, pd.to_datetime(refdate))] = fut_map

            # Hvis metric=corr og bruger ønsker min-corr, konverter score til corr (1-score) og filtrer
            if metric == 'corr' and args.min_corr is not None:
                # Score = 1 - corr -> corr = 1 - score
                df_scores['Corr'] = 1.0 - df_scores['Score']
                before = len(df_scores)
                df_scores = df_scores[df_scores['Corr'] >= args.min_corr].copy()
                after = len(df_scores)
                if after < before:
                    log.info("min-corr filter: fjernet %d af %d matches (threshold=%.3f)", before - after, before, args.min_corr)
                # Re-trim topn hvis vi filtrerede
                df_scores = df_scores.head(topn)
            elif metric == 'corr':
                df_scores['Corr'] = 1.0 - df_scores['Score']
            all_scores.append(df_scores.assign(Lookback=lookback, Horizon=horizon, Metric=metric,
                                               MatchCols=",".join(match_cols)))

            # Gem scores for denne query
            out_scores = out_dir / f"{_safe_filename(q_ticker)}_scores.csv"
            if args.dry_run:
                log.info("dry-run: would save scores to %s", out_scores)
            else:
                df_scores.to_csv(out_scores, index=False)
            status.step = "save_scores"
            status.last_msg = f"saved scores to {out_scores}"
            log.info("Scores gemt for %s: %s", q_ticker, out_scores)

            # Pak futures (langt format) for match_cols
            fut_rows = []
            for _, m_tkr, rdt, _, _ in top:
                key = (q_ticker, m_tkr, pd.to_datetime(rdt))
                fmap = futures.get(key, {})
                max_len = max((len(arr) for arr in fmap.values()), default=0)
                for i in range(max_len):
                    row = {"QueryTicker": q_ticker, "MatchTicker": m_tkr, "RefDate": pd.to_datetime(rdt),
                           "FutureOffset": i + 1}
                    for col in match_cols:
                        arr = fmap.get(col, np.array([]))
                        row[col] = float(arr[i]) if i < len(arr) and np.isfinite(arr[i]) else np.nan
                    fut_rows.append(row)
            df_future = pd.DataFrame(fut_rows).sort_values(["RefDate", "FutureOffset"]).reset_index(drop=True)

            # Enrich futures with metadata columns for symmetry with scores
            if not df_future.empty:
                df_future["Lookback"] = lookback
                df_future["Horizon"] = horizon
                df_future["MatchCols"] = ",".join(match_cols)
            out_future = out_dir / f"{_safe_filename(q_ticker)}_futures.parquet"
            if args.dry_run:
                log.info("dry-run: would save futures to %s", out_future)
            else:
                df_future.to_parquet(out_future, index=False)
            status.step = "save_futures"
            status.last_msg = f"saved futures to {out_future}"
            log.info("Futures gemt for %s: %s", q_ticker, out_future)

            # Plot (viser første kolonne i match_cols)
            if args.plot:
                try:
                    plot_matches(query_vecs, df_future.rename(columns={"MatchTicker": "MatchTicker"}), topk=min(topn, 10),
                                 plot_col=match_cols[0])
                except Exception as e:
                    log.error("Plot fejl for %s: %s", q_ticker, e)
        finally:
            # Mark this ticker done for progress reporting and update EWMA of per-ticker time
            try:
                dur = time.perf_counter() - ticker_start
                alpha = 0.3
                if status.ewma_seconds is None:
                    status.ewma_seconds = dur
                else:
                    status.ewma_seconds = alpha * dur + (1.0 - alpha) * status.ewma_seconds
            except Exception:
                dur = None
            try:
                status.processed = min(status.processed + 1, status.total)
            except Exception:
                status.processed += 1
            # update last_msg with last duration for visibility
            if dur is not None:
                status.last_msg = f"last_ticker={dur:.1f}s"

    # Gem samlet resume af scores (alle tickere)
    if all_scores:
        status.step = "save_all"
        status.last_msg = "saving aggregated scores"
        df_all = pd.concat(all_scores, ignore_index=True)
        out_all = out_dir / "all_scores.csv"
        if args.dry_run:
            log.info("dry-run: would save aggregated scores to %s", out_all)
        else:
            df_all.to_csv(out_all, index=False)
            log.info("Samlet score-oversigt gemt: %s", out_all)

    # Stop heartbeat thread if running
    if hb_thread is not None:
        hb_stop.set()
        hb_thread.join(timeout=2.0)
    # Ensure final heartbeat shows completed count
    status.processed = status.total
    status.step = "done"
    status.last_msg = "finished"
    if args.cache_ref:
        log.info("Cache stats: entries=%d capacity=%d", len(file_cache_order), MAX_CACHE)
    # Metadata (best effort)
    try:
        write_metadata(out_dir.as_posix(), name='analog', args=args, extra={'tickers': status.total})
    except Exception:
        pass
    log.info("completed: processed %d/%d", status.processed, status.total)


if __name__ == "__main__":
    main()
