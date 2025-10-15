"""Run pipeline for `watch.csv` (single input CSV).

Usage example (PowerShell):
    python run_watch_pipeline.py --input watch.csv --fetch-out data/fetch_watch --features-out data/features_watch --rebased-out data/rebased_watch --flat-out data/flat_vectors_watch
"""
from __future__ import annotations
import argparse
import os
import sys

proj_root = os.path.dirname(__file__)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.run_pipeline_common import run_full_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run full pipeline for watch.csv-like input")
    p.add_argument("--input", default="watch.csv", help="CSV med tickers")
    p.add_argument("--fetch-out", default="data/watch/fetch_watch", help="Output for fetch stage")
    p.add_argument("--features-out", default="data/watch/features_watch", help="Output for features stage")
    p.add_argument("--rebased-out", default="data/watch/rebased_watch", help="Output for rebased stage")
    p.add_argument("--flat-out", default="data/watch/flat_vectors_watch", help="Output for flat_vectorize stage")
    p.add_argument("--out-base", default=None, help="Optional base folder to derive all output folders from. Individual --*-out flags override this base when explicitly set.")

    # pass-through flags
    p.add_argument("--preset", choices=["standard", "fast", "validate"], default="standard")
    p.add_argument("--csv-head", type=int, default=1000)
    p.add_argument("--csv-tail", type=int, default=1000)
    p.add_argument("--float-dp", type=int, default=4)
    p.add_argument("--format", choices=["csv", "parquet", "both"], default="both")
    p.add_argument("--per-ticker", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands but don't execute pipeline stages")

    # fetch-specific
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--actions", action="store_true")

    # rebased-specific
    p.add_argument("--before", type=int, default=20)
    p.add_argument("--after", type=int, default=0)
    p.add_argument("--drop-rows-if-nan", choices=["none","any","all"], default="any")
    p.add_argument("--require-full-window", action="store_true")

    # flat_vectorize-specific
    p.add_argument("--all-refdates", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    import sys

    fetch_out = args.fetch_out
    features_out = args.features_out
    rebased_out = args.rebased_out
    flat_out = args.flat_out

    if args.out_base:
        if "--fetch-out" not in sys.argv:
            fetch_out = os.path.join(args.out_base, "fetch_watch")
        if "--features-out" not in sys.argv:
            features_out = os.path.join(args.out_base, "features_watch")
        if "--rebased-out" not in sys.argv:
            rebased_out = os.path.join(args.out_base, "rebased_watch")
        if "--flat-out" not in sys.argv:
            flat_out = os.path.join(args.out_base, "flat_vectors_watch")

    if not getattr(args, "dry_run", False):
        os.makedirs(fetch_out, exist_ok=True)
        os.makedirs(features_out, exist_ok=True)
        os.makedirs(rebased_out, exist_ok=True)
        os.makedirs(flat_out, exist_ok=True)
    run_full_pipeline(args.input, fetch_out, features_out, rebased_out, flat_out, args)


if __name__ == "__main__":
    main()
