"""Benchmark different matching pipelines (full scan, index prefilter, vectorized scoring).

Scenarier / modes:
    full               -> Baseline: gennemløb alle referencevinduer og score (loop).
    index               -> Prefilter vha. signatur-index (loop scoring af kandidater).
    index-vectorized    -> Prefilter + vectoriseret (batch) scoring (+ valgfri parallel threads).
    all (default)       -> Kør alle ovenstående i rækkefølge.

Understøtter både MSE og Corr metric. Corr-prefilter forsøger at bruge z-normaliserede signaturer
('<col>_sig_z') hvis de findes i index; ellers fallback til centrering.

Output: konsol + (valgfrit) JSON fil med tider og nøgletal (--json-out).
"""
from pathlib import Path
import time
import json
import argparse
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

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
    p.add_argument('--ref-dir', default='rebased', help='Mappe med rebased *_rebased.parquet filer')
    p.add_argument('--index', default='rebased_index.parquet', help='Signatur index parquet')
    p.add_argument('--match-cols', default='AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased')
    p.add_argument('--lookback', type=int, default=75)
    p.add_argument('--downsample', type=int, default=8, help='Downsample størrelse anvendt i index')
    p.add_argument('--topk', type=int, default=300, help='Prefilter topK kandidater')
    p.add_argument('--ticker', default=None, help='Query ticker (default første i watch.csv)')
    p.add_argument('--mode', default='all', choices=['full','index','index-vectorized','all'], help='Benchmark mode')
    p.add_argument('--metric', default='mse', choices=['mse','corr'], help='Scoring metric')
    p.add_argument('--prefilter-metric', default=None, choices=['mse','corr'], help='Metric brugt i prefilter (default = samme som --metric)')
    p.add_argument('--score-workers', type=int, default=1, help='Antal worker threads for vectoriseret scoring')
    p.add_argument('--json-out', default=None, help='Gem resultater som JSON fil')
    return p.parse_args()


def _load_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _coerce_sig_column(col_series: pd.Series) -> np.ndarray:
    def _to_array(j):
        if isinstance(j, str):
            try:
                arr = json.loads(j)
            except Exception:
                arr = []
            return np.array(arr, dtype=float)
        if isinstance(j, (list, tuple, np.ndarray)):
            return np.array(j, dtype=float)
        try:
            return np.array(j, dtype=float)
        except Exception:
            return np.array([], dtype=float)
    arrs = col_series.apply(_to_array).to_numpy()
    return np.vstack(arrs)


def compute_prefilter_distances(idx: pd.DataFrame, match_cols: List[str], query_vecs: Dict[str, np.ndarray], downsample: int, metric: str) -> np.ndarray:
    # Build (N,d) arrays for each column
    dists = None
    for col in match_cols:
        base_col = f'{col}_sig'
        use_corr = (metric == 'corr')
        have_z = use_corr and f'{col}_sig_z' in idx.columns
        if use_corr and have_z:
            A = _coerce_sig_column(idx[f'{col}_sig_z'])  # already z-normalized
            q_ds = np.array(downsample_series(query_vecs[col], downsample), dtype=float)
            # z-normalize query signature
            q_m = np.nanmean(q_ds)
            q_s = np.nanstd(q_ds)
            if not np.isfinite(q_s) or q_s == 0:
                # fallback: treat as zeros -> all distances ~1
                q_z = np.zeros_like(q_ds)
            else:
                q_z = (q_ds - q_m) / q_s
            # distance = 1 - mean(A * q_z)
            prod = A * q_z
            col_dist = 1.0 - np.nanmean(prod, axis=1)
        else:
            A = _coerce_sig_column(idx[base_col])
            q = np.array(downsample_series(query_vecs[col], downsample), dtype=float)
            if metric == 'mse':
                diff = A - q
                col_dist = np.nanmean(diff ** 2, axis=1)
            else:  # corr fallback via centered correlation
                A_c = A - np.nanmean(A, axis=1, keepdims=True)
                q_c = q - np.nanmean(q)
                denom = np.sqrt(np.nansum(A_c ** 2, axis=1) * np.nansum(q_c ** 2))
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = (A_c @ q_c) / denom
                corr = np.clip(corr, -1, 1)
                corr[~np.isfinite(corr)] = 0.0
                col_dist = 1.0 - corr
        if dists is None:
            dists = col_dist
        else:
            dists = dists + col_dist
    return dists / len(match_cols)


def load_candidate_windows(idx: pd.DataFrame, rows: np.ndarray, ref_dir: Path, match_cols: List[str], lookback: int) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, Any, str]]]:
    # Return dict col -> (N,L) arrays + metadata list (Ticker, RefDate, SourceFile) for windows actually loaded
    by_file = {}
    for i in rows:
        r = idx.iloc[i]
        by_file.setdefault(r['SourceFile'], []).append((r['Ticker'], r['RefDate']))
    col_arrays = {c: [] for c in match_cols}
    meta = []
    lb = lookback
    for sf, pairs in by_file.items():
        fp = ref_dir / sf
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        df['RefDate'] = pd.to_datetime(df['RefDate'], errors='coerce')
        for tkr, rdt in pairs:
            sel = df[(df['Ticker'] == tkr) & (df['RefDate'] == pd.to_datetime(rdt))]
            if sel.empty:
                continue
            g = sel.sort_values('Offset')
            # Expect -lookback+1 .. 0 -> length lookback
            win_ok = True
            per_col_vals = {}
            for col in match_cols:
                if col not in g.columns:
                    win_ok = False
                    break
                arr = pd.to_numeric(g[col], errors='coerce').to_numpy()
                if len(arr) != lb or np.isnan(arr).any():
                    win_ok = False
                    break
                per_col_vals[col] = arr
            if not win_ok:
                continue
            for col in match_cols:
                col_arrays[col].append(per_col_vals[col])
            meta.append((tkr, pd.to_datetime(rdt), sf))
    for col in match_cols:
        if col_arrays[col]:
            col_arrays[col] = np.vstack(col_arrays[col])
        else:
            col_arrays[col] = np.empty((0, lookback))
    return col_arrays, meta


def vectorized_score(query_vecs: Dict[str, np.ndarray], cand_arrays: Dict[str, np.ndarray], metric: str) -> np.ndarray:
    N = next(iter(cand_arrays.values())).shape[0]
    scores = np.zeros(N, dtype=float)
    for col, qv in query_vecs.items():
        A = cand_arrays[col]  # (N,L)
        if metric == 'mse':
            diff = A - qv
            scores += np.mean(diff * diff, axis=1)
        else:
            A_c = A - A.mean(axis=1, keepdims=True)
            q_c = qv - qv.mean()
            denom = np.sqrt((A_c * A_c).sum(axis=1) * (q_c * q_c).sum())
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = (A_c @ q_c) / denom
            corr = np.clip(corr, -1, 1)
            corr[~np.isfinite(corr)] = 0.0
            scores += (1.0 - corr)
    return scores / len(query_vecs)


def vectorized_score_parallel(query_vecs: Dict[str, np.ndarray], cand_arrays: Dict[str, np.ndarray], metric: str, workers: int) -> np.ndarray:
    if workers <= 1:
        return vectorized_score(query_vecs, cand_arrays, metric)
    N = next(iter(cand_arrays.values())).shape[0]
    if N == 0:
        return np.empty(0)
    # Choose chunk count = workers (bounded by N)
    workers = min(workers, N)
    idx_splits = np.array_split(np.arange(N), workers)
    def _chunk_score(ix):
        sub = {c: arr[ix] for c, arr in cand_arrays.items()}
        return ix, vectorized_score(query_vecs, sub, metric)
    out = np.zeros(N, dtype=float)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ix, sc in ex.map(_chunk_score, idx_splits):
            out[ix] = sc
    return out


def run_full_scan(ref_dir: Path, match_cols: List[str], lookback: int, query_vecs: Dict[str, np.ndarray], metric: str):
    start = time.time()
    n = 0
    for fname, r_ticker, refdate, win_map, fut_map in iter_reference_windows(str(ref_dir), match_cols, lookback, 1):
        try:
            _ = score_window_simple(query_vecs, win_map, metric=metric)
        except Exception:
            continue
        n += 1
    elapsed = time.time() - start
    return {
        'windows': n,
        'time': elapsed,
        'windows_per_sec': n / elapsed if elapsed > 0 else None
    }


def run_index_prefilter(idx: pd.DataFrame, ref_dir: Path, match_cols: List[str], query_vecs: Dict[str, np.ndarray], lookback: int, downsample: int, topk: int, metric: str, prefilter_metric: str):
    # distances
    start = time.time()
    dists = compute_prefilter_distances(idx, match_cols, query_vecs, downsample, prefilter_metric)
    order = np.argsort(dists)
    sel = order[:topk]
    t_pref = time.time() - start
    # loop scoring
    start2 = time.time()
    scored = 0
    for i in sel:
        r = idx.iloc[i]
        fp = ref_dir / r['SourceFile']
        try:
            win = load_window_from_file(fp, r['Ticker'], r['RefDate'], lookback, match_cols)
        except Exception:
            continue
        try:
            _ = score_window_simple(query_vecs, win, metric=metric)
        except Exception:
            continue
        scored += 1
    t_score = time.time() - start2
    return {
        'prefilter_time': t_pref,
        'prefilter_rows': int(len(dists)),
        'topk': int(len(sel)),
        'candidate_scoring_time': t_score,
        'scored_candidates': scored
    }


def run_index_vectorized(idx: pd.DataFrame, ref_dir: Path, match_cols: List[str], query_vecs: Dict[str, np.ndarray], lookback: int, downsample: int, topk: int, metric: str, prefilter_metric: str, workers: int):
    start = time.time()
    dists = compute_prefilter_distances(idx, match_cols, query_vecs, downsample, prefilter_metric)
    order = np.argsort(dists)
    sel = order[:topk]
    t_pref = time.time() - start
    # load candidate windows into arrays
    load_start = time.time()
    cand_arrays, meta = load_candidate_windows(idx, sel, ref_dir, match_cols, lookback)
    t_load = time.time() - load_start
    score_start = time.time()
    scores = vectorized_score_parallel(query_vecs, cand_arrays, metric, workers=workers) if meta else np.empty(0)
    t_score = time.time() - score_start
    return {
        'prefilter_time': t_pref,
        'prefilter_rows': int(len(dists)),
        'topk': int(len(sel)),
        'loaded_candidates': int(len(meta)),
        'load_time': t_load,
        'candidate_scoring_time': t_score,
        'scored_candidates': int(len(scores)),
        'vectorized': True,
        'workers': workers
    }


def main():
    args = parse_args()
    match_cols = [c.strip() for c in args.match_cols.split(',') if c.strip()]
    ref_dir = Path(args.ref_dir)
    prefilter_metric = args.prefilter_metric or args.metric

    # pick query ticker
    if args.ticker:
        ticker = args.ticker
    else:
        watch = pd.read_csv('watch.csv')
        ticker = str(watch['Ticker'].dropna().iloc[0]).strip()

    print(f'Query ticker: {ticker}  lookback={args.lookback} metric={args.metric} prefilter_metric={prefilter_metric}')
    query_vecs = load_query_from_rebased(ref_dir, ticker, args.lookback, match_cols)

    results = {}
    modes = [args.mode] if args.mode != 'all' else ['full','index','index-vectorized']
    idx_df = None
    for mode in modes:
        if mode == 'full':
            print('\n[MODE full]')
            res = run_full_scan(ref_dir, match_cols, args.lookback, query_vecs, metric=args.metric)
            print(f"Full scan: windows={res['windows']} time={res['time']:.2f}s rate={res['windows_per_sec']:.1f}/s")
            results['full'] = res
        elif mode in ('index','index-vectorized'):
            if idx_df is None:
                print('Indlæser index ...')
                idx_df = _load_index(Path(args.index))
            if mode == 'index':
                print('\n[MODE index]')
                res = run_index_prefilter(idx_df, ref_dir, match_cols, query_vecs, args.lookback, args.downsample, args.topk, args.metric, prefilter_metric)
                print(f"Prefilter: rows={res['prefilter_rows']} topk={res['topk']} time={res['prefilter_time']:.2f}s")
                print(f"Scoring (loop): scored={res['scored_candidates']} time={res['candidate_scoring_time']:.2f}s")
                results['index'] = res
            else:
                print('\n[MODE index-vectorized]')
                res = run_index_vectorized(idx_df, ref_dir, match_cols, query_vecs, args.lookback, args.downsample, args.topk, args.metric, prefilter_metric, args.score_workers)
                print(f"Prefilter: rows={res['prefilter_rows']} topk={res['topk']} time={res['prefilter_time']:.2f}s")
                print(f"Load candidates: loaded={res['loaded_candidates']} time={res['load_time']:.2f}s")
                print(f"Scoring (vectorized workers={res['workers']}): scored={res['scored_candidates']} time={res['candidate_scoring_time']:.2f}s")
                results['index_vectorized'] = res

    # simple derived speedups
    speedup_iv = None
    speedup_i = None
    if 'full' in results and 'index_vectorized' in results:
        a = results['full']['time']
        b = results['index_vectorized']['prefilter_time'] + results['index_vectorized']['load_time'] + results['index_vectorized']['candidate_scoring_time']
        if a > 0 and b > 0:
            speedup_iv = a / b
            print(f"\nSpeedup full -> index-vectorized: {speedup_iv:.2f}x")
    if 'full' in results and 'index' in results:
        a = results['full']['time']
        b = results['index']['prefilter_time'] + results['index']['candidate_scoring_time']
        if a > 0 and b > 0:
            speedup_i = a / b
            print(f"Speedup full -> index: {speedup_i:.2f}x")

    payload = {
        'metric': args.metric,
        'prefilter_metric': prefilter_metric,
        'query_ticker': ticker,
        'results': results,
        'speedup_full_to_index': speedup_i,
        'speedup_full_to_index_vectorized': speedup_iv,
        'mode': args.mode,
        'topk': args.topk,
        'downsample': args.downsample,
        'score_workers': args.score_workers
    }

    ts_path = None
    if args.json_out:
        try:
            with open(args.json_out, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            print(f'Resultater skrevet til {args.json_out}')
        except Exception as e:
            print('Kunne ikke skrive JSON:', e)
    else:
        # auto snapshot
        snap_root = Path('analog_out/benchmark_runs')
        snap_root.mkdir(parents=True, exist_ok=True)
        import datetime as _dt
        ts = _dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        ts_path = snap_root / f'bench_{ts}.json'
        with open(ts_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f'Auto snapshot skrevet til {ts_path}')


if __name__ == '__main__':
    main()
