import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Ensure project root on sys.path when running file directly from an editor
proj_root = os.path.dirname(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from utils.common import round_for_csv, write_metadata
import logging
logger = logging.getLogger("aktier.rebased")


# ---------------------------
# CLI / preset handling
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Lav rebaset vinduer [-before,+after] handelsdage for hver ref-dag")

    # Use SUPPRESS so we can detect if a user actually provided a flag
    p.add_argument("--input", "-i", help="Input CSV eller Parquet (fil eller mappe) med features", default=argparse.SUPPRESS)
    p.add_argument("--out", "-o", help="Output mappe", default=argparse.SUPPRESS)
    p.add_argument("--per-ticker", action="store_true", help="Gem én fil per ticker i stedet for samlet fil", default=argparse.SUPPRESS)

    p.add_argument("--before", type=int, help="Antal dage før (default=50)", default=argparse.SUPPRESS)
    p.add_argument("--after", type=int, help="Antal dage efter (default=50)", default=argparse.SUPPRESS)

    p.add_argument("--float-dp", type=int, help="Antal decimaler i CSV (Parquet bevarer fuld præcision)", default=argparse.SUPPRESS)
    p.add_argument("--format", choices=["csv", "parquet", "both"], help="Output format", default=argparse.SUPPRESS)

    p.add_argument(
        "--drop-rows-if-nan",
        choices=["none", "any", "all"],
        help=(
            "Styr rækker med NaN i rebased-kolonner: "
            "none=behold alle rækker; any=drop rækker hvor nogen rebased-kolonne er NaN; "
            "all=drop kun rækker hvor alle rebased-kolonner er NaN."
        ),
        default=argparse.SUPPRESS,
    )
    p.add_argument("--csv-head", type=int, help="Hvis sat: skriv kun de første N rækker til samplet CSV", default=argparse.SUPPRESS)
    p.add_argument("--csv-tail", type=int, help="Hvis sat: skriv også de sidste N rækker til samplet CSV", default=argparse.SUPPRESS)

    p.add_argument(
        "--require-full-window",
        action="store_true",
        help=(
            "Kræv at hele vinduet [-before,+after] er komplet efter NaN-filtrering. "
            "Hvis ikke, springes vinduet/ref-dagen over. Giver første komplette datasæt fra offset -before."
        ),
        default=argparse.SUPPRESS,
    )

    p.add_argument("--preset", choices=["standard"], help="Forudindstillet kørsel: standard", default=argparse.SUPPRESS)
    p.add_argument("--show-config", action="store_true", help="Print den endelige konfiguration (efter preset) som JSON og exit.", default=argparse.SUPPRESS)
    p.add_argument("--config-out", help="Hvis sat: skriv den resolved konfiguration som JSON til denne fil og exit.", default=argparse.SUPPRESS)
    p.add_argument("--log-file", help="JSON-lines logfil (append).", default=argparse.SUPPRESS)
    p.add_argument("--debug", action="store_true", help="Print debug info about existing outputs og config matching.", default=argparse.SUPPRESS)

    return p.parse_args()


def _set_default(ns, key, value):
    """Only set default if the user didn't provide the key."""
    if key not in vars(ns):
        setattr(ns, key, value)


def apply_preset(args):
    # If run without any CLI args (e.g., VS Code "Run Python File"), default to preset=standard
    if "preset" not in vars(args) and len(sys.argv) == 1:
        args.preset = "standard"

    if getattr(args, "preset", None) == "standard":
        # standard forventer compute_features output
        if "input" not in vars(args):
            default_dir = os.path.join("data", "features")
            default_file = os.path.join(default_dir, "features.parquet")
            if os.path.isdir(default_dir):
                args.input = default_dir
            elif os.path.isfile(default_file):
                args.input = default_file
            else:
                args.input = default_dir

        _set_default(args, "out", os.path.join("data", "rebased"))
        _set_default(args, "before", 20)
        _set_default(args, "after", 5)
        _set_default(args, "per_ticker", False)
        _set_default(args, "float_dp", 4)
        _set_default(args, "format", "both")
        _set_default(args, "drop_rows_if_nan", "any")
        _set_default(args, "require_full_window", True)
        _set_default(args, "csv_head", 1000)
        _set_default(args, "csv_tail", 1000)

    else:
        _set_default(args, "before", 50)
        _set_default(args, "after", 50)
        _set_default(args, "format", "csv")
        _set_default(args, "drop_rows_if_nan", "all")
        _set_default(args, "require_full_window", False)

    return args


# ---------------------------
# IO helpers / metadata
# ---------------------------

def _normalize_paths(args):
    """Normalize relative paths to project root for out/input/config_out."""
    try:
        for attr in ("out", "input", "config_out"):
            if getattr(args, attr, None):
                val = getattr(args, attr)
                if not os.path.isabs(val):
                    setattr(args, attr, os.path.abspath(os.path.join(proj_root, val)))
    except Exception:
        pass


def _sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".meta.json")


def _write_sidecar(parquet_path: Path, config: dict):
    try:
        meta = {
            "file": str(parquet_path.name),
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "config": config,
        }
        with open(_sidecar_path(parquet_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def _read_sidecar(parquet_path: Path) -> dict | None:
    p = _sidecar_path(parquet_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("config") if isinstance(meta, dict) else None
    except Exception:
        return None


def _read_meta_config(folder: str | Path) -> dict | None:
    """Backwards-compatible helper expected by tests: read a folder-level
    `_meta.json` (written by older pipeline) and return the `config` dict.
    """
    try:
        p = Path(folder) / "_meta.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("config") if isinstance(meta, dict) else None
    except Exception:
        return None


def _config_matches(existing_cfg: dict | None, args: object, keys: list[str]) -> bool:
    if not existing_cfg:
        return False
    try:
        args_cfg = {k: getattr(args, k, None) for k in keys}
    except Exception:
        return False

    def _norm(v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    for k in keys:
        if _norm(existing_cfg.get(k)) != _norm(args_cfg.get(k)):
            return False
    return True


def _append_missing_rows_to_parquet(out_pq: Path, new_df: pd.DataFrame, key_cols: list[str]) -> int:
    """If out_pq exists, read it and append only rows from new_df whose key_cols are missing.
    Returns number of appended rows. If parquet doesn't exist or incompatible, returns -1 to signal full rewrite.
    """
    try:
        if not out_pq.exists():
            return -1
        existing = pd.read_parquet(out_pq)
        # quick compatibility check: same columns
        if set(existing.columns) != set(new_df.columns):
            return -1
        existing_keys = existing[key_cols].drop_duplicates()
        merged = new_df.merge(existing_keys, on=key_cols, how="left", indicator=True)
        new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
        if new_rows.empty:
            return 0
        appended = pd.concat([existing, new_rows], ignore_index=True)
        appended = order_columns(appended)
        appended.to_parquet(out_pq, index=False, compression="snappy")
        return len(new_rows)
    except Exception:
        return -1


# -------- CSV upsert helpers --------

def _upsert_full_csv(csv_path: Path, new_df: pd.DataFrame, key_cols: list[str], float_dp: int | None):
    """
    Opret/udvid en FULD CSV ved at merge på key_cols og droppe dubletter.
    Bevarer kolonneorden via order_columns.
    """
    try:
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            # Kolonne-kompatibilitet: hvis afviger, reskriver vi fuldt fra new_df
            if set(existing.columns) == set(new_df.columns):
                merged = pd.concat([existing, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=key_cols, keep="first")
            else:
                merged = new_df.copy()
        else:
            merged = new_df.copy()

        merged = order_columns(merged)
        num_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
        df_csv, csv_float_format = round_for_csv(merged, float_dp, include_cols=num_cols)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_csv.to_csv(csv_path, index=False, float_format=csv_float_format)
        return len(merged)
    except Exception as e:
        print(f"CSV upsert fejl for {csv_path}: {e}")
        # Fallback: skriv kun new_df
        num_cols = [c for c in new_df.columns if pd.api.types.is_numeric_dtype(new_df[c])]
        df_csv, csv_float_format = round_for_csv(new_df, float_dp, include_cols=num_cols)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_csv.to_csv(csv_path, index=False, float_format=csv_float_format)
        return len(new_df)


def _write_sampled_csv_from_full(full_csv: Path, sampled_csv: Path, head: int, tail: int):
    """
    Laver samplet CSV (head/tail) ud fra FULD CSV.
    Hvis head+tail >= N, bliver sampled = full.
    """
    if head is None and tail is None:
        return  # intet at gøre

    if not full_csv.exists():
        return  # intet at sample fra

    df_full = pd.read_csv(full_csv)
    n = len(df_full)
    h = max(0, head or 0)
    t = max(0, tail or 0)
    if h + t >= n:
        df_sample = df_full
    else:
        idx_keep = list(range(h)) + list(range(max(0, n - t), n))
        df_sample = df_full.iloc[idx_keep]

    sampled_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(sampled_csv, index=False)


# ---------------------------
# Data loading
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)

    if p.is_dir():
        # Prefer parquet files if present, otherwise CSV
        pq_files = sorted(p.glob("*.parquet"))
        if pq_files:
            parts = [pd.read_parquet(f) for f in pq_files]
            df = pd.concat(parts, ignore_index=True)
        else:
            csv_files = sorted(p.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"Ingen .parquet eller .csv filer i mappen: {p}")
            parts = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(parts, ignore_index=True)
    else:
        # Single file or parquet dataset directory (pandas can read dir-like parquet too)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)

    # Validate required columns
    needed = {"Ticker", "Date"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Mangler kolonner: {sorted(missing)}")

    # Parse and clean dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Sort and dedupe
    df = (
        df.sort_values(["Ticker", "Date"])
        .drop_duplicates(subset=["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )

    # Ensure numeric types for value columns
    ma_cols = [c for c in df.columns if c.startswith("MA_")]
    value_cols = [c for c in ["AdjClose", "Close"] if c in df.columns] + ma_cols
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------
# Core computation
# ---------------------------

def _iterative_make_windows_optimized(
    g: pd.DataFrame,
    before: int,
    after: int,
    drop_policy: str = "all",
    require_full_window: bool = False,
) -> pd.DataFrame:
    """Optimized iterative version that reduces pandas __setitem__ overhead by
    building column arrays in a dict and creating one DataFrame per window.
    This is a low-risk improvement over the original implementation.
    """
    ma_cols = [c for c in g.columns if c.startswith("MA_")]
    value_cols = [c for c in g.columns if c in ["AdjClose", "Close"]] + ma_cols
    rsi_cols = [c for c in g.columns if c.upper().startswith("RSI")]

    rows_out = []
    n = len(g)
    win_len = before + after + 1

    ticker_val = g["Ticker"].iloc[0]

    for center in range(before, n - after):
        ref_row = g.iloc[center]
        ref_date = ref_row["Date"]
        ref_vals = {c: ref_row.get(c, pd.NA) for c in value_cols}

        def _is_valid_ref(v):
            if pd.isna(v):
                return False
            try:
                return bool(v != 0)
            except Exception:
                return False

        valid_ref_cols = [c for c, v in ref_vals.items() if _is_valid_ref(v)]
        if not valid_ref_cols:
            continue

        window = g.iloc[center - before : center + after + 1]
        offsets = np.arange(-before, after + 1, dtype=int)

        # build arrays first, then one DataFrame construction
        out_dict = {}
        out_dict["Ticker"] = np.full(win_len, ticker_val)
        out_dict["RefDate"] = np.full(win_len, ref_date)
        out_dict["Offset"] = offsets
        out_dict["Date"] = window["Date"].to_numpy()

        # rebased value columns
        for c in value_cols:
            col_name = f"{c}_Rebased"
            if c in valid_ref_cols:
                base = ref_vals[c]
                arr = window[c].to_numpy(dtype=float)
                out_dict[col_name] = (arr / base) * 100.0
            else:
                out_dict[col_name] = np.full(win_len, np.nan)

        # RSI and other passthroughs
        for c in rsi_cols:
            out_dict[c] = window[c].to_numpy()

        # filtering using numpy for speed
        rebased_cols = [f"{c}_Rebased" for c in value_cols]
        if not rebased_cols:
            mask_keep = np.ones(win_len, dtype=bool)
        else:
            stack = np.column_stack([out_dict[col].astype(float) for col in rebased_cols])
            if drop_policy == "none":
                mask_keep = np.ones(win_len, dtype=bool)
            elif drop_policy == "any":
                mask_keep = ~np.isnan(stack).any(axis=1)
            else:  # all
                mask_keep = ~np.isnan(stack).all(axis=1)

        if require_full_window and not mask_keep.all():
            continue

        if not mask_keep.all():
            # apply mask to arrays
            for k in list(out_dict.keys()):
                out_dict[k] = out_dict[k][mask_keep]

        out = pd.DataFrame(out_dict)
        if not out.empty:
            rows_out.append(out)

    return pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()


def _vectorized_make_windows(
    g: pd.DataFrame,
    before: int,
    after: int,
    drop_policy: str = "all",
    require_full_window: bool = False,
) -> pd.DataFrame:
    """Vectorized implementation using numpy.sliding_window_view.
    Builds all windows at once and flattens to long format. This should be
    much faster for large series.
    """
    n = len(g)
    win_len = before + after + 1
    if n < win_len:
        return pd.DataFrame()

    centers = np.arange(before, n - after)
    num_windows = len(centers)
    start_idxs = centers - before

    # helper to build sliding windows for a column
    def _sw_for(col):
        arr = g[col].to_numpy()
        try:
            sw = sliding_window_view(arr, win_len)
        except Exception:
            # fallback: build with list comprehension (slower)
            sw = np.array([arr[i : i + win_len] for i in range(n - win_len + 1)])
        return sw[start_idxs]

    ma_cols = [c for c in g.columns if c.startswith("MA_")]
    value_cols = [c for c in g.columns if c in ["AdjClose", "Close"]] + ma_cols
    rsi_cols = [c for c in g.columns if c.upper().startswith("RSI")]

    # Ref dates per window
    ref_dates = g["Date"].to_numpy()[centers]

    offsets = np.arange(-before, after + 1, dtype=int)

    # build rebased arrays per value_col: shape (num_windows, win_len)
    rebased_map = {}
    any_valid_ref = False
    for c in value_cols:
        sw = _sw_for(c).astype(float)
        ref_vals = sw[:, before]
        valid_mask = ~np.isnan(ref_vals) & (ref_vals != 0)
        if valid_mask.any():
            any_valid_ref = True
        # avoid divide-by-zero warnings
        ref_vals_safe = ref_vals.copy()
        ref_vals_safe[~valid_mask] = np.nan
        reb = (sw / ref_vals_safe[:, None]) * 100.0
        # where ref invalid, set entire row to nan
        reb[~valid_mask, :] = np.nan
        rebased_map[f"{c}_Rebased"] = reb

    if not any_valid_ref:
        return pd.DataFrame()

    # dates windows
    date_sw = _sw_for("Date")

    # flatten arrays to long format
    flat = {}
    flat_size = num_windows * win_len
    flat["Ticker"] = np.repeat(g["Ticker"].iloc[0], flat_size)
    flat["RefDate"] = np.repeat(ref_dates, win_len)
    flat["Offset"] = np.tile(offsets, num_windows)
    flat["Date"] = date_sw.reshape(-1)

    for k, arr in rebased_map.items():
        flat[k] = arr.reshape(-1)

    for c in rsi_cols:
        sw = _sw_for(c)
        flat[c] = sw.reshape(-1)

    df_out = pd.DataFrame(flat)

    # filtering
    rebased_cols = list(rebased_map.keys())
    if not rebased_cols:
        mask_keep = np.ones(flat_size, dtype=bool)
    else:
        stack = np.column_stack([df_out[col].astype(float) for col in rebased_cols])
        if drop_policy == "none":
            mask_keep = ~np.zeros((flat_size,), dtype=bool)
        elif drop_policy == "any":
            mask_keep = ~np.isnan(stack).any(axis=1)
        else:
            mask_keep = ~np.isnan(stack).all(axis=1)

    if require_full_window:
        # compute per-window all-True and drop entire windows that fail
        mask_window = mask_keep.reshape(num_windows, win_len)
        full_ok = mask_window.all(axis=1)
        # build final mask by repeating per-window flag
        final_mask = np.repeat(full_ok, win_len)
        df_out = df_out.loc[final_mask].reset_index(drop=True)
    else:
        df_out = df_out.loc[mask_keep].reset_index(drop=True)

    return df_out


def make_windows_for_ticker(
    g: pd.DataFrame,
    before: int,
    after: int,
    drop_policy: str = "all",
    require_full_window: bool = False,
) -> pd.DataFrame:
    """Wrapper that tries vectorized implementation first and falls back to
    the optimized iterative version on failure.
    """
    try:
        return _vectorized_make_windows(g, before, after, drop_policy, require_full_window)
    except Exception:
        # if any error in vectorized path, fallback to safer iterative optimized path
        return _iterative_make_windows_optimized(g, before, after, drop_policy, require_full_window)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Fast orden: Ticker, RefDate, Offset, Date, *_Rebased (AdjClose, Close, MA_* sorteret numerisk), RSI_*
    base = ["Ticker", "RefDate", "Offset", "Date"]
    rebased_fixed = ["AdjClose_Rebased", "Close_Rebased"]

    def _num_sfx(name: str, prefix: str) -> float:
        try:
            return float(name[len(prefix) :])
        except Exception:
            return float("inf")

    ma_rebased = [c for c in df.columns if c.startswith("MA_") and c.endswith("_Rebased")]
    # strip _Rebased to read the numeric part
    ma_sorted = sorted(ma_rebased, key=lambda c: _num_sfx(c[:-9], "MA_"))
    rebased_cols = [c for c in rebased_fixed if c in df.columns] + ma_sorted

    rsi_cols = [c for c in df.columns if c.upper().startswith("RSI")]
    rsi_sorted = sorted(rsi_cols, key=lambda c: _num_sfx(c.upper(), "RSI_"))

    rest = [c for c in df.columns if c not in base and c not in rebased_cols and c not in rsi_sorted]
    cols = [c for c in base if c in df.columns] + rebased_cols + rsi_sorted + rest
    return df[cols]


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    args = apply_preset(args)
    _normalize_paths(args)

    # initialise logging handlers (console + optional JSON file)
    try:
        from utils.log import init_logging
        init_logging(getattr(args, "log_file", None))
    except Exception:
        pass

    # If user requested to only show the resolved config, print and exit before any I/O
    if getattr(args, "show_config", False):
        cfg = json.dumps(vars(args), default=str, sort_keys=True, ensure_ascii=False, indent=2)
        print(cfg)
        if getattr(args, "config_out", None):
            with open(args.config_out, "w", encoding="utf-8") as _f:
                _f.write(cfg)
        return

    if not getattr(args, "input", None):
        raise SystemExit("Fejl: --input mangler. Angiv --input eller brug --preset standard.")
    if not getattr(args, "out", None):
        raise SystemExit("Fejl: --out mangler. Angiv --out eller brug --preset standard.")

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.input)
    all_out = []

    # Keys to guard compatibility
    guard_keys = ["before", "after", "drop_rows_if_nan", "require_full_window"]

    for ticker, g in df.groupby("Ticker", sort=False, observed=False):
        print(f"Processing {ticker} ({len(g)} rows)...")
        rebased = make_windows_for_ticker(
            g,
            getattr(args, "before", 50),
            getattr(args, "after", 50),
            drop_policy=getattr(args, "drop_rows_if_nan", "all"),
            require_full_window=getattr(args, "require_full_window", False),
        )
        if rebased.empty:
            continue

        rebased = order_columns(rebased)

        if getattr(args, "per_ticker", False):
            # --- Parquet canonical ---
            if getattr(args, "format", "csv") in ("parquet", "both"):
                out_pq = Path(args.out) / f"{ticker}_rebased.parquet"
                out_pq.parent.mkdir(parents=True, exist_ok=True)

                # check sidecar compatibility
                cfg_ok = _config_matches(_read_sidecar(out_pq), args, guard_keys)
                appended_flag = False
                if cfg_ok:
                    appended = _append_missing_rows_to_parquet(out_pq, rebased, key_cols=["Ticker", "RefDate", "Offset"])
                    if appended == -1:
                        rebased.to_parquet(out_pq, index=False, compression="snappy")
                        _write_sidecar(out_pq, {k: getattr(args, k, None) for k in guard_keys})
                        print(f"  -> saved Parquet {len(rebased)} rows to {out_pq}")
                    else:
                        appended_flag = True
                        if appended == 0:
                            print(f"  -> no new rows to append to {out_pq}")
                        else:
                            print(f"  -> appended {appended} rows to existing {out_pq}")
                else:
                    rebased.to_parquet(out_pq, index=False, compression="snappy")
                    _write_sidecar(out_pq, {k: getattr(args, k, None) for k in guard_keys})
                    print(f"  -> saved Parquet {len(rebased)} rows to {out_pq}")

            # --- CSV (full + sampled) ---
            if getattr(args, "format", "csv") in ("csv", "both"):
                # 1) FULD CSV append/upsert
                full_csv = Path(args.out) / f"{ticker}_rebased.csv"
                _upsert_full_csv(full_csv, rebased, key_cols=["Ticker", "RefDate", "Offset"], float_dp=getattr(args, "float_dp", None))

                # 2) Samplet CSV afledt af fuld CSV
                h = getattr(args, "csv_head", None)
                t = getattr(args, "csv_tail", None)
                if h is not None or t is not None:
                    sampled_csv = Path(args.out) / f"{ticker}_rebased.sampled.csv"
                    _write_sampled_csv_from_full(full_csv, sampled_csv, h, t)
                    print(f"  -> updated sampled CSV at {sampled_csv}")
                else:
                    print(f"  -> updated full CSV at {full_csv}")

                if getattr(args, "debug", False):
                    print("DEBUG: per-ticker full_csv=", full_csv)

        else:
            all_out.append(rebased)

    # Combined output branch
    if not getattr(args, "per_ticker", False) and all_out:
        combined = order_columns(pd.concat(all_out, ignore_index=True))

        # --- Parquet canonical ---
        if getattr(args, "format", "csv") in ("parquet", "both"):
            out_pq = Path(args.out) / "rebased_all.parquet"
            out_pq.parent.mkdir(parents=True, exist_ok=True)
            cfg_ok = _config_matches(_read_sidecar(out_pq), args, guard_keys)

            if cfg_ok:
                appended = _append_missing_rows_to_parquet(out_pq, combined, key_cols=["Ticker", "RefDate", "Offset"])
                if appended == -1:
                    combined.to_parquet(out_pq, index=False, compression="snappy")
                    _write_sidecar(out_pq, {k: getattr(args, k, None) for k in guard_keys})
                    print(f"Saved combined Parquet with {len(combined)} rows to {out_pq}")
                elif appended == 0:
                    print(f"No new rows to append to {out_pq}")
                else:
                    print(f"Appended {appended} rows to existing {out_pq}")
            else:
                combined.to_parquet(out_pq, index=False, compression="snappy")
                _write_sidecar(out_pq, {k: getattr(args, k, None) for k in guard_keys})
                print(f"Saved combined Parquet with {len(combined)} rows to {out_pq}")

        # --- CSV (full + sampled) ---
        if getattr(args, "format", "csv") in ("csv", "both"):
            full_csv = Path(args.out) / "rebased_all.csv"
            _upsert_full_csv(full_csv, combined, key_cols=["Ticker", "RefDate", "Offset"], float_dp=getattr(args, "float_dp", None))

            h = getattr(args, "csv_head", None)
            t = getattr(args, "csv_tail", None)
            if h is not None or t is not None:
                sampled_csv = Path(args.out) / "rebased_all.sampled.csv"
                _write_sampled_csv_from_full(full_csv, sampled_csv, h, t)
                print(f"Saved sampled combined CSV at {sampled_csv}")
            else:
                print(f"Saved full combined CSV at {full_csv}")

            if getattr(args, "debug", False):
                print("DEBUG: combined full_csv=", full_csv)
                print("DEBUG: parquet exists=",
                      (Path(args.out) / "rebased_all.parquet").exists() if getattr(args, "format", "csv") in ("parquet", "both") else "n/a")

    # Folder-level metadata (backwards-compatible med din eksisterende pipeline)
    try:
        write_metadata(
            getattr(args, "out", None),
            name="rebased",
            args=args,
            extra={"before": getattr(args, "before", None), "after": getattr(args, "after", None)},
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
