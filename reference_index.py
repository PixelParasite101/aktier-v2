"""Build a lightweight signature index for rebased windows.

Usage (example):

  python reference_index.py --rebased-dir rebased --out rebased_index.parquet \
      --match-cols AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased \
      --lookback 75 --horizon 25 --sig-downsample 8

The index stores one row per (Ticker, RefDate, SourceFile) window that has a full valid lookback segment.
Each signature column <col>_sig is a fixed-length list[float] of length sig_downsample.

We downsample using simple linear sampling of the lookback window (not including future offsets).

Design goals:
- Fast load (single parquet) instead of scanning all rebased parquet files for prefiltering.
- Keep schema stable; additional metadata columns can be added later without breaking existing usage.

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import sys
import logging
from utils.common import write_metadata
logger = logging.getLogger("aktier.index")


def parse_cols_list(s: str) -> List[str]:
    return [c.strip() for c in s.split(',') if c.strip()]


def downsample(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if arr.size == 0:
        return np.full(n, np.nan, dtype=float)
    idx = np.linspace(0, max(0, len(arr) - 1), num=n, dtype=int)
    return arr[idx].astype(float, copy=False)


def build_index(rebased_dir: Path, out: Path, match_cols: List[str], lookback: int, sig_downsample: int) -> pd.DataFrame:
    files = sorted(rebased_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"Ingen parquet filer i {rebased_dir}")

    lb_start, lb_end = -lookback + 1, 0
    rows = []
    for fp in files:
        try:
            # Only load needed columns to reduce IO
            needed = ['Ticker', 'RefDate', 'Offset'] + match_cols
            df = pd.read_parquet(fp, columns=needed)
        except Exception:
            df = pd.read_parquet(fp)
            df = df[[c for c in df.columns if c in needed]]
        missing = set(['Ticker', 'RefDate', 'Offset']) - set(df.columns)
        if missing:
            continue
        # Filter lookback offsets once
        df_lb_all = df[(df['Offset'] >= lb_start) & (df['Offset'] <= lb_end)]
        if df_lb_all.empty:
            continue
        # Group by each window reference (Ticker, RefDate)
        for (ticker, refdate), g in df_lb_all.groupby(['Ticker', 'RefDate'], sort=False):
            g = g.sort_values('Offset')
            # Validate full lookback length
            if len(g) != lookback:
                continue
            ok = True
            sigs = {}
            col_stats = {}
            for col in match_cols:
                if col not in g.columns:
                    ok = False
                    break
                vals = pd.to_numeric(g[col], errors='coerce').to_numpy()
                if np.isnan(vals).any() or len(vals) != lookback:
                    ok = False
                    break
                ds = downsample(vals, sig_downsample)
                sigs[f'{col}_sig'] = ds.tolist()
                col_mean = float(vals.mean())
                col_std = float(vals.std(ddof=0))
                if col_std == 0:
                    col_std = float('nan')
                col_stats[f'{col}_mean'] = col_mean
                col_stats[f'{col}_std'] = col_std
                # z-normaliseret signatur for corr prefilter
                if np.isfinite(col_std) and col_std != 0:
                    z = (ds - col_mean) / col_std
                else:
                    z = np.full_like(ds, np.nan)
                sigs[f'{col}_sig_z'] = z.tolist()
            if not ok:
                continue
            row = {
                'Ticker': ticker,
                'RefDate': pd.to_datetime(refdate),
                'SourceFile': fp.name,
                'Lookback': lookback,
                'SigSize': sig_downsample,
            }
            row.update(sigs)
            row.update(col_stats)
            rows.append(row)
    if not rows:
        raise RuntimeError('Ingen gyldige vinduer fundet til index.')
    df_index = pd.DataFrame(rows)
    # Sort for determinism
    df_index = df_index.sort_values(['Ticker', 'RefDate']).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_parquet(out, index=False)
    return df_index


def _flag_provided(*names: str) -> bool:
    return any(n in sys.argv for n in names)


def apply_preset(args):
    if not args.preset:
        return args
    if args.preset == "standard":
        # match fetch/features/rebased konvention: auto defaults hvis user ikke har sat
        if args.rebased_dir == 'rebased' and not _flag_provided('--rebased-dir'):
            args.rebased_dir = 'rebased'
        if args.out == 'rebased_index.parquet' and not _flag_provided('--out'):
            args.out = 'rebased_index.parquet'
        if args.match_cols == 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased' and not _flag_provided('--match-cols'):
            args.match_cols = 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased'
        if args.lookback == 75 and not _flag_provided('--lookback'):
            args.lookback = 75
        if args.sig_downsample == 8 and not _flag_provided('--sig-downsample'):
            args.sig_downsample = 8
    return args


def main():
    ap = argparse.ArgumentParser(description='Build signature index for rebased windows.')
    ap.add_argument('--rebased-dir', default='rebased', help='Directory containing *_rebased.parquet files')
    ap.add_argument('--out', default='rebased_index.parquet', help='Output parquet file for index')
    ap.add_argument('--match-cols', default='AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased')
    ap.add_argument('--lookback', type=int, default=75)
    ap.add_argument('--sig-downsample', type=int, default=8)
    ap.add_argument('--preset', choices=['standard'], default=None, help='Preset konfiguration (pt. kun standard).')
    ap.add_argument('--show-config', action='store_true', help='Vis resolved config som JSON og exit.')
    ap.add_argument('--config-out', default=None, help='Skriv resolved config til fil og exit.')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Auto preset hvis scriptet køres direkte uden argumenter for konsistens
    if args.preset is None and len(sys.argv) == 1:
        args.preset = 'standard'
    args = apply_preset(args)

    if getattr(args, 'show_config', False):
        cfg_json = json.dumps(vars(args), default=str, sort_keys=True, ensure_ascii=False, indent=2)
        print(cfg_json)
        if args.config_out:
            with open(args.config_out, 'w', encoding='utf-8') as f:
                f.write(cfg_json)
        return

    match_cols = parse_cols_list(args.match_cols)
    rebased_dir = Path(args.rebased_dir)
    out = Path(args.out)

    df = build_index(rebased_dir, out, match_cols, args.lookback, args.sig_downsample)
    # Metadata (best effort) -> skriv i en dedikeret mappe ved siden af output filen (fx rebased_index_meta/)
    try:
        meta_dir = out.parent / (out.stem + "_meta")
        write_metadata(meta_dir.as_posix(), name='index', args=args, extra={'rows': int(len(df)), 'index_file': out.name})
    except Exception:
        pass
    if not args.quiet:
        logger.info('Index rows: %d; columns: %s', len(df), list(df.columns))
        logger.info('Saved to %s', out)


if __name__ == '__main__':
    main()
