"""Benchmark full-scan vs index-prefilter approach for analog matcher.

Produces timing and candidate counts for a single query ticker (default from watch.csv).
"""
from pathlib import Path
import time
import json
import argparse
from typing import List

import numpy as np
import pandas as pd

# Reuse helper from analog_matcher_watch by importing functions
from pathlib import Path as _Path


# Inline minimal iter_reference_windows and build_query_vectors to avoid import issues
def build_query_vectors(df, lookback: int, match_cols):
    if len(df) < lookback:
        raise ValueError(f"Kun {len(df)} rækker hentet; kræver lookback={lookback}.")
    tail = df.tail(lookback).reset_index(drop=True)
    vecs = {}
    for col in match_cols:
        if col not in tail.columns:
            raise ValueError(f"Query mangler kolonnen '{col}'.")
        v = pd.to_numeric(tail[col], errors='coerce').to_numpy()
        if np.isnan(v).any():
            raise ValueError(f"Query kolonnen '{col}' indeholder NaN i de sidste {lookback} dage.")
        vecs[col] = v
    return vecs


def iter_reference_windows(ref_dir: str, match_cols: List[str], lookback: int, horizon: int):
    d = _Path(ref_dir)
    files = sorted(d.glob('*.parquet'))
    lb_start, lb_end = -lookback + 1, 0
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=['Ticker','RefDate','Offset'] + match_cols)
        except Exception:
            df = pd.read_parquet(fp)
        # subset
        needed = {'Ticker','RefDate','Offset'} | set(match_cols)
        if not needed.issubset(set(df.columns)):
            continue
        df_lb = df[(df['Offset'] >= lb_start) & (df['Offset'] <= lb_end)]
        if df_lb.empty:
            continue
        for (ticker, refdate), g in df_lb.groupby(['Ticker','RefDate'], sort=False):
            g = g.sort_values('Offset')
            win_map = {}
            ok = True
            for col in match_cols:
                s = pd.to_numeric(g[col], errors='coerce').to_numpy()
                if len(s) != lookback or np.isnan(s).any():
                    ok = False
                    break
                win_map[col] = s
            if not ok:
                continue
            fut_map = {}
            for col in match_cols:
                fut = df[(df['Ticker']==ticker) & (df['RefDate']==refdate) & (df['Offset']>=1) & (df['Offset']<=horizon)]
                arr = pd.to_numeric(fut[col], errors='coerce').to_numpy()
                fut_map[col] = arr
            yield (fp.name, ticker, pd.to_datetime(refdate), win_map, fut_map)


def downsample_series(x: np.ndarray, n: int) -> List[float]:
    if n <= 0:
        return []
    if len(x) == 0:
        return [float('nan')] * n
    idx = np.linspace(0, max(0, len(x) - 1), num=n, dtype=int)
    out = []
    for i in idx:
        v = x[i]
        out.append(float(v) if np.isfinite(v) else float('nan'))
    return out


def load_query_from_rebased(ref_dir: Path, ticker: str, lookback: int, match_cols: List[str]):
    # pick the most recent RefDate for ticker from its rebased parquet
    p = ref_dir / f"{ticker}_rebased.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p, columns=['Ticker','RefDate','Offset'] + match_cols)
    # find the latest RefDate
    df['RefDate'] = pd.to_datetime(df['RefDate'], errors='coerce')
    latest = df['RefDate'].max()
    lb_start = -lookback + 1
    lb_end = 0
    win = df[(df['RefDate'] == latest) & (df['Offset'] >= lb_start) & (df['Offset'] <= lb_end)].sort_values('Offset')
    if win.empty:
        raise RuntimeError('No window found for query')
    vecs = {}
    for col in match_cols:
        vecs[col] = pd.to_numeric(win[col], errors='coerce').to_numpy()
    return vecs


def score_window_simple(query_vecs, win_map, metric='mse'):
    # simple scoring used in script: average of per-col _score_pair
    total = 0.0
    cols = list(query_vecs.keys())
    for col in cols:
        a = query_vecs[col]
        b = win_map[col]
        if metric == 'mse':
            total += float(np.mean((a - b) ** 2))
        else:
            if np.std(a) == 0 or np.std(b) == 0:
                total += 1.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                total += 1.0 - corr
    return total / len(cols)


def load_window_from_file(fp: Path, ticker: str, refdate, lookback: int, match_cols: List[str]):
    df = pd.read_parquet(fp)
    df = df[(df['Ticker'] == ticker) & (pd.to_datetime(df['RefDate']) == pd.to_datetime(refdate))]
    df = df.sort_values('Offset')
    win_map = {}
    for col in match_cols:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors='coerce').to_numpy()
        else:
            arr = np.array([])
        win_map[col] = arr
    return win_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-dir', default='rebased')
    p.add_argument('--index', default='rebased_index.parquet')
    p.add_argument('--match-cols', default='AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased')
    p.add_argument('--lookback', type=int, default=75)
    p.add_argument('--downsample', type=int, default=8)
    p.add_argument('--topk', type=int, default=300)
    p.add_argument('--ticker', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    match_cols = [c.strip() for c in args.match_cols.split(',') if c.strip()]
    ref_dir = Path(args.ref_dir)
    idx = pd.read_parquet(args.index)

    # pick query ticker: either provided or first from watch.csv
    if args.ticker:
        ticker = args.ticker
    else:
        watch = pd.read_csv('watch.csv')
        ticker = str(watch['Ticker'].dropna().iloc[0]).strip()

    print('Building query vectors from rebased for', ticker)
    query_vecs = load_query_from_rebased(ref_dir, ticker, args.lookback, match_cols)

    # FULL SCAN
    start = time.time()
    n = 0
    best_full = []
    for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(str(ref_dir), match_cols, args.lookback, 1):
        try:
            s = score_window_simple(query_vecs, win_map, metric='mse')
        except Exception:
            continue
        best_full.append((r_ticker, refdate, s, fname))
        n += 1
    t_full = time.time() - start
    print(f'Full scan: processed {n} windows in {t_full:.2f}s. average {n/t_full:.1f} windows/s')

    # PREFILTER using index
    # build signature for query
    q_sigs = []
    for col in match_cols:
        arr = query_vecs[col]
        q_sigs.append(np.array(downsample_series(arr, args.downsample), dtype=float))
    # load index signatures into matrix
    idx_sigs = []
    for col in match_cols:
        # support both old JSON-string signatures and new native list/ndarray signatures
        def _to_array(j):
            if isinstance(j, str):
                try:
                    arr = json.loads(j)
                except Exception:
                    # fallback: treat as single NaN-filled signature
                    arr = []
                return np.array(arr, dtype=float)
            # pandas may read list/array columns as numpy.ndarray or list
            if isinstance(j, (list, tuple)):
                return np.array(j, dtype=float)
            try:
                return np.array(j, dtype=float)
            except Exception:
                return np.array([], dtype=float)

        col_sig = idx[f'{col}_sig'].apply(_to_array)
        idx_sigs.append(np.vstack(col_sig.to_numpy()))
    # compute simple distance (MSE) across signatures
    start = time.time()
    # stack per-col distances
    dists = None
    for col_idx in range(len(match_cols)):
        A = idx_sigs[col_idx]  # (N, d)
        q = q_sigs[col_idx]
        # compute per-row mse, ignoring NaNs by masking
        diff = A - q
        mse = np.nanmean(diff ** 2, axis=1)
        if dists is None:
            dists = mse
        else:
            dists = dists + mse
    # average
    dists = dists / len(match_cols)
    topk_idx = np.argsort(dists)[:args.topk]
    t_prefilter = time.time() - start
    print(f'Prefilter step: computed distances for {len(dists)} index rows in {t_prefilter:.2f}s, selected top {len(topk_idx)}')

    # Now do full scoring only for topk candidates
    start = time.time()
    candidates = []
    # index has Ticker,RefDate,SourceFile
    for i in topk_idx:
        row = idx.iloc[i]
        candidates.append((row['Ticker'], row['RefDate'], row['SourceFile']))

    full_candidates = 0
    best_pref = []
    # group candidates by source file so we only read each file once
    by_file = {}
    for tkr, rdt, sf in candidates:
        by_file.setdefault(sf, []).append((tkr, rdt))
    for sf, pairs in by_file.items():
        fp = ref_dir / sf
        df = pd.read_parquet(fp)
        for tkr, rdt in pairs:
            g = df[(df['Ticker'] == tkr) & (pd.to_datetime(df['RefDate']) == pd.to_datetime(rdt))]
            g = g.sort_values('Offset')
            win_map = {}
            for col in match_cols:
                win_map[col] = pd.to_numeric(g[col], errors='coerce').to_numpy()
            try:
                s = score_window_simple(query_vecs, win_map, metric='mse')
            except Exception:
                continue
            best_pref.append((tkr, rdt, s, sf))
            full_candidates += 1
    t_scoring = time.time() - start
    print(f'Prefilter full scoring: scored {full_candidates} candidates in {t_scoring:.2f}s')

    # Summary
    print('SUMMARY:')
    print(f'  full_scan_time={t_full:.2f}s  windows={n}')
    print(f'  prefilter_time={t_prefilter:.2f}s  topk={len(topk_idx)}  scoring_time={t_scoring:.2f}s')


if __name__ == '__main__':
    main()
