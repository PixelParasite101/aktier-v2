"""Run pipeline for `aktie.csv` (single input CSV).

Usage example (PowerShell):
    python run_aktie_pipeline.py --input aktie.csv --fetch-out data/fetch --features-out data/features --rebased-out data/rebased --flat-out data/flat_vectors

The script accepts a subset of flags and forwards them to each stage.
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
    p = argparse.ArgumentParser(description="Run full pipeline for aktie.csv-like input")
    p.add_argument("--input", default="aktie.csv", help="CSV med tickers")
    p.add_argument("--fetch-out", default="data/aktie/fetch_data_all", help="Output for fetch stage")
    p.add_argument("--features-out", default="data/aktie/features", help="Output for features stage")
    p.add_argument("--rebased-out", default="data/aktie/rebased", help="Output for rebased stage")
    p.add_argument("--flat-out", default="data/aktie/flat_vectors", help="Output for flat_vectorize stage")
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
    p.add_argument("--after", type=int, default=5)
    p.add_argument("--drop-rows-if-nan", choices=["none","any","all"], default="any")
    p.add_argument("--require-full-window", action="store_true")

    # flat_vectorize-specific
    p.add_argument("--all-refdates", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    # If --out-base is provided, derive outputs from it unless the user explicitly
    # provided per-output flags. We detect explicit flags using sys.argv.
    import sys

    fetch_out = args.fetch_out
    features_out = args.features_out
    rebased_out = args.rebased_out
    flat_out = args.flat_out

    if args.out_base:
        if "--fetch-out" not in sys.argv:
            fetch_out = os.path.join(args.out_base, "fetch_data_all")
        if "--features-out" not in sys.argv:
            features_out = os.path.join(args.out_base, "features")
        if "--rebased-out" not in sys.argv:
            rebased_out = os.path.join(args.out_base, "rebased")
        if "--flat-out" not in sys.argv:
            flat_out = os.path.join(args.out_base, "flat_vectors")

    if not getattr(args, "dry_run", False):
        os.makedirs(fetch_out, exist_ok=True)
        os.makedirs(features_out, exist_ok=True)
        os.makedirs(rebased_out, exist_ok=True)
        os.makedirs(flat_out, exist_ok=True)
    run_full_pipeline(args.input, fetch_out, features_out, rebased_out, flat_out, args)


if __name__ == "__main__":
    main()
