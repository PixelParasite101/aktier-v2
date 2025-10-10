# Copilot instructions for aktier-v2

This repo is a small, test-backed pipeline for daily stock data: fetch -> features -> rebased windows -> flat vectors -> watch matching. Scripts default to sensible presets and write both CSV (rounded) and Parquet (full precision) with metadata.

## Big picture
- Data flow
  1) src/fetch_history_pro.py: yfinance daily 1d OHLCV (+dividends/splits) -> data/fetch_data_all/*
  2) src/compute_features.py: add MA_N and RSI_L -> data/features/features.parquet(+csv)
  3) src/make_rebased_windows.py: build per-(Ticker, RefDate) windows with rebased columns (…_Rebased) -> data/rebased/*
  4) src/flat_vectorize.py: flatten per-window features to vectors -> data/flat_vectors/*
  5) src/compare_watch_with_flat_vectors.py: compute a query vector for watch.csv ticker and rank matches against flat vectors
- Utilities: utils/common.py (round_for_csv, write_metadata), utils/log.py (console + optional JSON-lines)
- Tests assume specific column names/order; keep naming and order helpers intact.

## Conventions that matter
- CSV rounding only: use utils.common.round_for_csv(float_dp) for CSV; keep full precision for Parquet.
- Required columns by stage: fetch -> {Ticker, Date, Open, High, Low, Close, AdjClose, Volume}. Rebased adds RefDate, Offset, and *_Rebased columns.
- Column order helpers: compute_features.order_columns, make_rebased_windows.order_columns; tests rely on these.
- Presets: Each CLI supports --preset standard with smart defaults and --show-config/--config-out to resolve final args without running.
- Metadata: write_metadata(out_dir, name=…) creates _meta.json with git hash and config. Rebased parquet files also get sidecar .meta.json for incremental appends.
- Logging: pass --log-file to append JSON-lines; console logs remain human-readable.

## Typical workflows (PowerShell)
- Fetch (defaults to preset=standard when no args):
  - python -m src.fetch_history_pro
  - Or explicitly: python -m src.fetch_history_pro --preset standard --use-watch
- Features:
  - python -m src.compute_features --preset standard
- Rebased windows (vectorized path with fallback):
  - python -m src.make_rebased_windows --preset standard
- Flat vectors (all groups; auto-detect window from Offsets):
  - python -m src.flat_vectorize --preset standard --all-refdates
- Watch matching (writes per-ticker outputs under data/watch_flat_vectors/<TICKER>/):
  - python -m src.compare_watch_with_flat_vectors
- Utility index over rebased windows:
  - python scripts/build_rebased_index.py --ref-dir data/rebased --out data/flat_vectors/rebased_index.parquet

## Project-specific patterns
- Incremental behavior:
  - Fetch: --incremental merges on (Ticker, Date); per-ticker parquet files are appended safely.
  - Rebased: parquet sidecars store config; code appends only missing (Ticker, RefDate, Offset) rows when config matches.
- NaN policy for rebased windows: --drop-rows-if-nan=none|any|all and --require-full-window; defaults in preset standard favor reliable, full windows.
- Feature naming: MA_<N>, RSI_<L>; rebased naming: <Base>_Rebased (AdjClose_Rebased, Close_Rebased, MA_<N>_Rebased).
- compare_watch auto-aligns the query vector length to dataset metadata (Window, Features) and exports the exact query used.

## Examples from repo
- Tests: see tests/test_smoke.py (offline E2E), tests/test_features.py, tests/test_rebased.py, tests/test_parquet_io_and_incremental.py.
- Data layout examples live in data/* (features.parquet, rebased_all.parquet/csv, flat_vectors.parquet/csv).

## When extending
- Keep parse_args/apply_preset + --show-config pattern consistent across CLIs.
- Preserve column names and use order_columns helpers to avoid test breakage.
- For CSV output, round via round_for_csv; keep original df for Parquet.
- If adding new indicators, follow MA_/RSI_ naming and include them in order_columns and rebasing logic where appropriate.

If any section is unclear or you need more details (e.g., preferred feature sets, window sizes, or additional scripts to document), tell me what to refine and I’ll update this file.
