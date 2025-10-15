"""Shared helper for running the pipeline stages for a single CSV input.

This module is intended to be used by the thin wrappers `run_aktie_pipeline.py`
and `run_watch_pipeline.py`. It calls the four stages as separate Python
module invocations using the current interpreter so the project's venv is used.

The design keeps things simple and explicit: each stage gets an input and out
path and shares a small set of common flags (csv_head/csv_tail/float_dp/format).
"""
from __future__ import annotations
import sys
import subprocess
from typing import List, Iterable

PY = sys.executable


def _run(cmd: List[str], dry_run: bool = False):
    print("Running:", " ".join(cmd))
    if dry_run:
        print("Dry-run: command not executed")
        return 0
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)} (rc={proc.returncode})")


def build_common_flags(args, allowed: Iterable[str]) -> List[str]:
    """Build flags from args restricted to allowed names (strings including leading dashes).

    Example: allowed could be {"--preset","--csv-head","--csv-tail","--float-dp","--format"}
    """
    flags: List[str] = []

    def _push(name: str, val: str | None):
        if name in allowed and val is not None:
            flags.extend([name, str(val)])

    _push("--preset", getattr(args, "preset", None))
    _push("--csv-head", getattr(args, "csv_head", None))
    _push("--csv-tail", getattr(args, "csv_tail", None))
    _push("--float-dp", getattr(args, "float_dp", None))
    _push("--format", getattr(args, "format", None))
    # per-ticker uses a flag without value
    if ("--per-ticker" in allowed) and (getattr(args, "per_ticker", False) or getattr(args, "per-ticker", False)):
        flags.append("--per-ticker")
    if ("--debug" in allowed) and getattr(args, "debug", False):
        flags.append("--debug")

    return flags


def run_full_pipeline(input_csv: str, fetch_out: str, features_out: str, rebased_out: str, flat_out: str, args) -> None:
    # Stage 1: fetch_history_pro
    fetch_allowed = {"--input", "--out", "--batch-size", "--incremental", "--actions", "--csv-head", "--csv-tail", "--float-dp", "--preset", "--validate-only", "--dry-run", "--log-file"}
    fetch_cmd = [PY, "-m", "src.fetch_history_pro", "--input", input_csv, "--out", fetch_out]
    fetch_cmd += build_common_flags(args, fetch_allowed)
    # stage-specific flags already filtered; run
    _run(fetch_cmd, dry_run=getattr(args, "dry_run", False))

    # Stage 2: compute_features
    # compute_features does not accept a --format flag, avoid forwarding it
    compute_allowed = {"--input", "--out", "--csv-head", "--csv-tail", "--float-dp", "--preset", "--per-ticker", "--debug"}
    features_cmd = [PY, "-m", "src.compute_features", "--input", fetch_out, "--out", features_out]
    # Ensure compute_features writes its CSV into the same features_out folder
    import os as _os
    # If features_out is a directory, make explicit output file paths inside it
    if _os.path.isdir(features_out) or not _os.path.splitext(features_out)[1]:
        out_parquet = _os.path.join(features_out, "features.parquet")
        csv_out = _os.path.join(features_out, "features.csv")
    else:
        # features_out is a file path
        out_parquet = features_out
        base = _os.path.splitext(features_out)[0]
        csv_out = base + ".csv"

    # Force compute_features to write a single parquet file (disable partitioning)
    features_cmd += build_common_flags(args, compute_allowed)
    # only add --csv if not already provided by caller
    if "--csv" not in features_cmd:
        features_cmd += ["--csv", csv_out]
    # override --out to be the explicit parquet file path
    # and explicitly disable partitioning by providing an empty value
    # for --partition-by so the preset won't set partitioning.
    features_cmd += ["--out", out_parquet, "--partition-by", ""]
    _run(features_cmd, dry_run=getattr(args, "dry_run", False))

    # Stage 3: make_rebased_windows
    rebased_allowed = {"--input", "--out", "--csv-head", "--csv-tail", "--float-dp", "--preset", "--format", "--before", "--after", "--drop-rows-if-nan", "--require-full-window", "--per-ticker", "--debug"}
    rebased_cmd = [PY, "-m", "src.make_rebased_windows", "--input", features_out, "--out", rebased_out]
    rebased_cmd += build_common_flags(args, rebased_allowed)
    if getattr(args, "before", None) is not None and "--before" not in rebased_cmd:
        rebased_cmd += ["--before", str(args.before)]
    if getattr(args, "after", None) is not None and "--after" not in rebased_cmd:
        rebased_cmd += ["--after", str(args.after)]
    if getattr(args, "drop_rows_if_nan", None) is not None and "--drop-rows-if-nan" not in rebased_cmd:
        rebased_cmd += ["--drop-rows-if-nan", args.drop_rows_if_nan]
    if getattr(args, "require_full_window", False) and "--require-full-window" not in rebased_cmd:
        rebased_cmd += ["--require-full-window"]
    _run(rebased_cmd, dry_run=getattr(args, "dry_run", False))

    # Stage 4: flat_vectorize
    # flat_vectorize expects --csv (input) not --input. Also it doesn't accept --format or --float-dp.
    flat_allowed = {"--csv-head", "--csv-tail", "--preset", "--all-refdates", "--per-ticker", "--debug"}
    # If rebased_out is a directory, pick the combined file written by make_rebased_windows
    import os as _os
    if _os.path.isdir(rebased_out):
        pq_candidate = _os.path.join(rebased_out, "rebased_all.parquet")
        csv_candidate = _os.path.join(rebased_out, "rebased_all.csv")
        if _os.path.exists(pq_candidate):
            csv_input = pq_candidate
        elif _os.path.exists(csv_candidate):
            csv_input = csv_candidate
        else:
            raise SystemExit(f"Could not find rebased_all.parquet or rebased_all.csv in {rebased_out}")
    else:
        csv_input = rebased_out

    # Run the script implementation in src directly (no top-level wrapper)
    import os as _os
    repo_root = _os.path.dirname(_os.path.dirname(__file__))
    src_flat = _os.path.join(repo_root, "src", "flat_vectorize.py")
    flat_cmd = [PY, src_flat, "--csv", csv_input, "--out", flat_out]
    flat_cmd += build_common_flags(args, flat_allowed)
    if getattr(args, "all_refdates", False) and "--all-refdates" not in flat_cmd:
        flat_cmd += ["--all-refdates"]
    _run(flat_cmd, dry_run=getattr(args, "dry_run", False))
