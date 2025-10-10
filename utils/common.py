"""Common helper utilities: rounding for CSV outputs and metadata writing.

This module centralizes small cross-script helpers to reduce duplication.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import pandas as pd


def round_for_csv(df: pd.DataFrame, float_dp: Optional[int], include_cols: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, Optional[str]]:
    """Return a (possibly) rounded copy of df suitable for CSV output and the float_format string.

    Only rounds numeric columns. If include_cols is provided, restrict rounding to that subset
    (columns silently ignored if missing). Parquet writers should keep original precision,
    so caller must keep original df for Parquet.
    """
    if float_dp is None:
        return df, None
    float_format = f"%.{float_dp}f"
    if df.empty:
        return df.copy(), float_format
    if include_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = [c for c in include_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df.copy(), float_format
    out = df.copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(float_dp)
    return out, float_format


def _get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def write_metadata(directory: str, name: str, args: Optional[object] = None, extra: Optional[dict] = None) -> str:
    """Write a small metadata JSON file into `directory`.

    name: logical component (e.g. 'fetch', 'features', 'rebased').
    args: typically the argparse.Namespace (will be converted via vars()).
    extra: any extra key/values to merge.

    Returns path to metadata file.
    """
    os.makedirs(directory, exist_ok=True)
    meta = {
        "component": name,
        # timezone-aware UTC timestamp with Z suffix
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": sys.version.split()[0],
        "pandas": getattr(pd, "__version__", None),
        "git_commit": _get_git_commit(),
    }
    if args is not None:
        try:
            meta["config"] = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        except Exception:
            pass
    if extra:
        meta.update(extra)
    path = os.path.join(directory, "_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

__all__ = ["round_for_csv", "write_metadata"]


def sample_and_round_df_to_csv(df, csv_path, head: int = 1000, tail: int = 1000, float_dp: int = 4):
    """Sample head+tail from a dataframe, round numeric columns, and write CSV.

    If df has <= head+tail rows the full dataframe is written. Returns path written.
    """
    from pathlib import Path

    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        # write empty CSV
        pd.DataFrame().to_csv(p, index=False)
        return str(p)

    n = len(df)
    h = max(0, head or 0)
    t = max(0, tail or 0)
    if n <= h + t:
        sample = df.copy()
    else:
        sample = pd.concat([df.head(h), df.tail(t)], ignore_index=True)

    numeric_cols = [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])]
    df_csv, csv_float_format = round_for_csv(sample, float_dp, include_cols=numeric_cols)
    # atomic write
    tmp = p.with_suffix(p.suffix + ".tmp")
    df_csv.to_csv(tmp, index=False, float_format=csv_float_format)
    try:
        tmp.replace(p)
    except Exception:
        # fallback
        df_csv.to_csv(p, index=False, float_format=csv_float_format)
    return str(p)


def sample_and_round_parquet_to_csv(parquet_path, csv_path, head: int = 1000, tail: int = 1000, float_dp: int = 4):
    """Read parquet, sample head+tail, round numeric columns and write CSV."""
    from pathlib import Path

    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(p)
    return sample_and_round_df_to_csv(df, csv_path, head=head, tail=tail, float_dp=float_dp)
