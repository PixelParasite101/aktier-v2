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
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


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
    return df[cols].reset_index(drop=True)


def compute_mas_inplace(df: pd.DataFrame) -> None:
    if "AdjClose" not in df.columns:
        raise ValueError("AdjClose mangler i hentede data.")
    df["MA_20"] = pd.to_numeric(df["AdjClose"], errors="coerce").rolling(20).mean()
    df["MA_50"] = pd.to_numeric(df["AdjClose"], errors="coerce").rolling(50).mean()
    df["MA_200"] = pd.to_numeric(df["AdjClose"], errors="coerce").rolling(200).mean()


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
        s = pd.to_numeric(df[base], errors="coerce")
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

def main():
    ap = argparse.ArgumentParser(description="Watch-list analog matching mod rebased reference-datasæt")
    ap.add_argument("--watch", required=True, help="CSV med kolonnen 'Ticker'")
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
    args = ap.parse_args()

    ref_dir = args.ref_dir
    lookback = args.lookback
    horizon = args.horizon
    topn = args.topn
    metric = args.metric
    match_cols = _parse_cols_list(args.match_cols)
    weights = _parse_weights_list(args.weights, len(match_cols)) if args.weights else [1.0] * len(match_cols)

    # Læs watch.csv
    watch_df = pd.read_csv(args.watch)
    if "Ticker" not in watch_df.columns:
        raise SystemExit("watch.csv skal have kolonnen 'Ticker'.")
    tickers = [str(t).strip() for t in watch_df["Ticker"].dropna().tolist() if str(t).strip()]
    if not tickers:
        raise SystemExit("watch.csv indeholder ingen tickere.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    for q_ticker in tickers:
        print(f"\n=== Query: {q_ticker} ===")
        # Byg query-data fra features (hvis valgt), ellers yfinance
        df: Optional[pd.DataFrame] = None
        if args.query_from_features:
            base_cols = [ _base_from_rebased_name(c) for c in match_cols ]
            df = _load_query_from_features(args.query_features_dir, q_ticker, base_cols)
            if df is None or df.empty:
                print(f"  Kunne ikke indlæse features for {q_ticker} fra {args.query_features_dir}; falder tilbage til yfinance…")

        if df is None:
            # Hent 200+lookback handelsdage (for at kunne beregne MA_200 korrekt)
            min_days = 200 + lookback
            df = fetch_stock_history_yf(q_ticker, min_trading_days=min_days)
            compute_mas_inplace(df)

        # Rebase de kolonner vi matcher på (baseret på sidste dato i datasættet)
        rebase_columns_inplace(df, cols_rebased=match_cols)

        # Byg query-vektorer (kun sidste lookback dage)
        query_vecs = build_query_vectors(df, lookback=lookback, match_cols=match_cols)

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
        for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(ref_dir, match_cols, lookback, horizon):
            try:
                s = weighted_score_window(query_vecs, win_map)
            except Exception:
                continue
            scores.append((q_ticker, r_ticker, refdate, s, fname))
            futures[(q_ticker, r_ticker, pd.to_datetime(refdate))] = fut_map

        if not scores:
            print(f"  Ingen gyldige referencevinduer fundet for {q_ticker}.")
            continue

        scores.sort(key=lambda x: x[3])  # lavest først
        top = scores[:topn]
        df_scores = pd.DataFrame(top, columns=["QueryTicker", "MatchTicker", "RefDate", "Score", "SourceFile"])
        all_scores.append(df_scores.assign(Lookback=lookback, Horizon=horizon, Metric=metric,
                                           MatchCols=",".join(match_cols)))

        # Gem scores for denne query
        out_scores = out_dir / f"{_safe_filename(q_ticker)}_scores.csv"
        df_scores.to_csv(out_scores, index=False)
        print(f"  Scores gemt: {out_scores}")

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

        out_future = out_dir / f"{_safe_filename(q_ticker)}_futures.parquet"
        df_future.to_parquet(out_future, index=False)
        print(f"  Futures gemt: {out_future}")

        # Plot (viser første kolonne i match_cols)
        if args.plot:
            try:
                plot_matches(query_vecs, df_future.rename(columns={"MatchTicker": "MatchTicker"}), topk=min(topn, 10),
                             plot_col=match_cols[0])
            except Exception as e:
                print(f"  Plot fejl: {e}")

    # Gem samlet resume af scores (alle tickere)
    if all_scores:
        df_all = pd.concat(all_scores, ignore_index=True)
        out_all = out_dir / "all_scores.csv"
        df_all.to_csv(out_all, index=False)
        print(f"\nSamlet score-oversigt gemt: {out_all}")


if __name__ == "__main__":
    main()
