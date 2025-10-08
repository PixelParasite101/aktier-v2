#!/usr/bin/env python3
"""
flat_vectorize.py (migrated to src)

Denne fil er flyttet ind i `src` for bedre struktur. Top-level `flat_vectorize.py` forbliver som wrapper.
"""
import os
import sys
# Ensure project root on sys.path when running file directly from an editor
# so imports like `from utils.common` resolve. When run as a module (python -m src.flat_vectorize)
# this is a no-op.
proj_root = os.path.dirname(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from numpy.lib.stride_tricks import sliding_window_view


class FlatVectorizeError(Exception):
    """Custom exception for flat_vectorize errors (helpers shouldn't sys.exit())."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=False, default=None, help="Sti til input CSV")
    p.add_argument("--out", required=False, default=None, help="Sti til output CSV")
    p.add_argument("--features", nargs="+", required=False, default=None,
                   help="Kolonner der bruges som features (rækkefølge = flad rækkefølge pr. dag)")
    p.add_argument("--window", type=int, default=75, help="Antal dage i vinduet (default: 75)")
    p.add_argument("--auto-window", action="store_true", help="Auto-detect window fra inputs Offset-range (max_offset - min_offset + 1)")
    p.add_argument("--ticker", help="Filtrér til én Ticker (valgfri)")
    p.add_argument("--refdate", help="Vælg én RefDate (YYYY-MM-DD) for single-vindue")
    p.add_argument("--all-refdates", action="store_true",
                   help="Generér vinduer for alle (Ticker, RefDate) grupper")
    p.add_argument("--require-consecutive", action="store_true",
                   help="Kræv at der findes præcis 'window' på hinanden følgende offsets fra min offset")
    p.add_argument("--scale", choices=["none", "zscore", "minmax"], default="none",
                   help="Skalering pr. vindue: none | zscore | minmax (default: none)")
    p.add_argument("--preset", choices=["standard"], default=None, help="Forudindstillet kørsel: standard")
    p.add_argument("--show-config", action="store_true", help="Print den endelige konfiguration (efter preset) som JSON og exit.")
    return p.parse_args()


def _flag_provided(*names: str) -> bool:
    import sys as _sys
    return any(n in _sys.argv for n in names)


def apply_preset(args):
    if not args.preset:
        return args
    if args.preset == "standard":
        # sensible defaults: read rebased_all, write to data/flat_vectors, common features
        if args.csv is None and not _flag_provided("--csv"):
            args.csv = os.path.join("data", "rebased", "rebased_all.csv")
        if args.out is None and not _flag_provided("--out"):
            args.out = os.path.join("data", "flat_vectors", "all_vectors.csv")
        if args.features is None and not _flag_provided("--features"):
            args.features = ["AdjClose_Rebased", "MA_20_Rebased", "RSI_14"]
        if not _flag_provided("--window"):
            args.window = 75
        # enable auto-window by default for the standard preset unless user specified otherwise
        if not _flag_provided("--auto-window"):
            args.auto_window = True
        if not _flag_provided("--all-refdates"):
            args.all_refdates = True
    return args

def check_columns(df: pd.DataFrame, needed):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"FEJL: Mangler kolonner i CSV: {missing}")

def select_group(df, ticker, refdate):
    g = df
    if ticker:
        g = g[g["Ticker"] == ticker]
    if refdate:
        g = g[g["RefDate"] == refdate]
    return g.copy()

def window_from_group(g: pd.DataFrame, features, window, require_consecutive=False):
    g = g.sort_values(["Offset", "Date"], ascending=[True, True])
    if require_consecutive:
        offsets = g["Offset"].to_numpy()
        offs0 = offsets - offsets.min()
        idx = None
        for start in range(0, len(offs0) - window + 1):
            segment = offs0[start:start+window]
            if np.array_equal(segment, np.arange(window)):
                idx = (start, start+window)
                break
        if idx is None:
            return None, 0
        cut = g.iloc[idx[0]:idx[1]]
    else:
        cut = g.head(window)

    if len(cut) < window:
        return None, len(cut)

    X = cut[features].to_numpy(dtype=float)  # [window, F]
    return X, window

def scale_window(X: np.ndarray, mode: str):
    if mode == "none":
        return X
    Xs = X.copy()
    if mode == "zscore":
        mu = Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, ddof=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (Xs - mu) / sd
    if mode == "minmax":
        x_min = Xs.min(axis=0, keepdims=True)
        x_max = Xs.max(axis=0, keepdims=True)
        rng = x_max - x_min
        rng[rng == 0] = 1.0
        return (Xs - x_min) / rng
    return X

def flatten_row(X: np.ndarray):
    return X.flatten()

def process_single(df, args):
    g = select_group(df, args.ticker, args.refdate)
    if g.empty:
        raise FlatVectorizeError("Ingen rækker matcher det valgte filter (--ticker/--refdate).")
    X, n = window_from_group(g, args.features, args.window, args.require_consecutive)
    if X is None:
        raise FlatVectorizeError(f"Kun {n} rækker i vindue – mindre end ønsket {args.window}. Prøv uden --require-consecutive eller brug kortere --window.")
    X = scale_window(X, args.scale)
    v = flatten_row(X)
    meta = {
        "Ticker": g["Ticker"].iloc[0],
        "RefDate": g["RefDate"].iloc[0],
        "Window": args.window,
        "Features": "|".join(args.features),
        "Scale": args.scale,
    }
    cols = [f"v{i}" for i in range(len(v))]
    out = pd.DataFrame([{**meta, **{c: v[i] for i, c in enumerate(cols)}}])
    out.to_csv(args.out, index=False)
    print(f"Skrev 1 vektor ({len(v)} dim) til {args.out}")

def process_all(df, args):
    out_df = collect_flat_vectors(df, args.features, args.window, args.require_consecutive, args.scale)
    if out_df is None or out_df.empty:
        sys.exit("Ingen grupper havde nok data til at danne et vindue.")
    out_df = out_df.sort_values(["Ticker", "RefDate"])
    out_df.to_csv(args.out, index=False)
    print(f"Skrev {len(out_df)} flade vektorer (hver på {args.window*len(args.features)} dim) til {args.out}")


def _vectorized_group_windows(g: pd.DataFrame, features, window: int):
    """Try to generate windows for a single group using numpy sliding_window_view.
    Only works reliably when offsets form a consecutive integer sequence (step=1).
    Returns list of 2D arrays (each window) or empty list.
    """
    # ensure sorted by Offset
    g = g.sort_values(["Offset", "Date"], ascending=[True, True])
    if "Offset" not in g.columns:
        return []
    offs = g["Offset"].to_numpy()
    # check consecutive with step 1
    if len(offs) < window:
        return []
    diffs = np.diff(offs)
    if not np.all(diffs == 1):
        return []

    # build sliding windows for features
    arrs = [g[f].to_numpy(dtype=float) for f in features]
    stacked = np.column_stack(arrs)  # shape (N, F)
    try:
        sw = sliding_window_view(stacked, (window, stacked.shape[1]))
        # sliding_window_view with 2D window returns shape (N-window+1, 1, window, F)
        # reshape to (num_windows, window, F)
        sw = sw[:, 0, :, :]
    except Exception:
        # fallback manual
        sw = np.array([stacked[i : i + window] for i in range(len(stacked) - window + 1)])

    return list(sw)


def collect_flat_vectors(df: pd.DataFrame, features, window: int, require_consecutive: bool, scale_mode: str):
    """Collect flattened vectors from all (Ticker, RefDate) groups and return DataFrame.
    Tries a vectorized path per group when offsets are consecutive; otherwise falls back to window_from_group.
    """
    rows = []
    group_cols = ["Ticker", "RefDate"]
    for (tic, rd), g in df.groupby(group_cols):
        # Attempt vectorized extraction when consecutive offsets are present and require_consecutive is True
        if require_consecutive:
            windows = _vectorized_group_windows(g, features, window)
            if not windows:
                continue
            # vectorized_group_windows returns all possible windows; keep first complete window to match original behavior
            X = windows[0]
            Xs = scale_window(X, scale_mode)
            v = flatten_row(Xs)
            row = {"Ticker": tic, "RefDate": rd, "Window": window, "Features": "|".join(features), "Scale": scale_mode}
            for i, val in enumerate(v):
                row[f"v{i}"] = val
            rows.append(row)
            continue

        # fallback behavior for non-require_consecutive or non-consecutive offsets: use existing window_from_group
        X, n = window_from_group(g, features, window, require_consecutive)
        if X is None:
            continue
        Xs = scale_window(X, scale_mode)
        v = flatten_row(Xs)
        row = {"Ticker": tic, "RefDate": rd, "Window": window, "Features": "|".join(features), "Scale": scale_mode}
        for i, val in enumerate(v):
            row[f"v{i}"] = val
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    import sys as _sys
    # If run without any CLI args (e.g., VS Code "Run Python File"), default to preset=standard
    if args.preset is None and len(_sys.argv) == 1:
        args.preset = "standard"
    args = apply_preset(args)
    # If user requested to only show the resolved config, print and exit before any I/O
    if getattr(args, "show_config", False):
        import json as _json
        cfg = _json.dumps(vars(args), default=str, sort_keys=True, ensure_ascii=False, indent=2)
        print(cfg)
        if getattr(args, "config_out", None):
            with open(args.config_out, "w", encoding="utf-8") as _f:
                _f.write(cfg)
        return

    if not args.csv:
        raise SystemExit("Fejl: --csv mangler. Angiv --csv eller brug --preset standard.")
    if not args.out:
        raise SystemExit("Fejl: --out mangler. Angiv --out eller brug --preset standard.")
    if not args.features:
        raise SystemExit("Fejl: --features mangler. Angiv --features eller brug --preset standard.")
    # Prefer parquet input if present (same base name with .parquet), fallback to CSV
    input_path = args.csv
    parquet_candidate = None
    try:
        from pathlib import Path
        p = Path(input_path)
        if p.suffix.lower() == ".parquet":
            parquet_candidate = p
        else:
            pq = p.with_suffix('.parquet')
            if pq.exists():
                parquet_candidate = pq
    except Exception:
        parquet_candidate = None

    if parquet_candidate is not None:
        df = pd.read_parquet(parquet_candidate)
    else:
        df = pd.read_csv(args.csv)
    check_columns(df, ["Ticker", "RefDate", "Offset", "Date"])
    check_columns(df, args.features)

    # Ensure feature columns are numeric (coerce if needed)
    for f in args.features:
        try:
            df[f] = pd.to_numeric(df[f], errors="coerce")
        except Exception:
            raise FlatVectorizeError(f"Feature column {f} could not be converted to numeric")

    # Auto-detect window from offset range if requested
    if args.auto_window:
        if "Offset" in df.columns:
            min_off = int(df["Offset"].min())
            max_off = int(df["Offset"].max())
            detected = max_off - min_off + 1
            args.window = int(detected)
            print(f"Auto-detected window={args.window} from offsets range [{min_off},{max_off}]")
        else:
            print("WARN: Offset-kolonne mangler; kan ikke auto-detecte window.")

    if args.all_refdates:
        out_df = collect_flat_vectors(df, args.features, args.window, args.require_consecutive, args.scale)
        if out_df is None or out_df.empty:
            raise FlatVectorizeError("Ingen grupper havde nok data til at danne et vindue.")

        # write parquet full
        out_path = args.out
        out_p = Path(out_path)
        if out_p.suffix == '':
            # treat as directory
            out_p_dir = out_p
            out_p_dir.mkdir(parents=True, exist_ok=True)
            pq_path = out_p_dir / "flat_vectors.parquet"
            csv_path = out_p_dir / "flat_vectors.csv"
        else:
            pq_path = out_p.with_suffix('.parquet')
            csv_path = out_p.with_suffix('.csv')
            out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_df.to_parquet(pq_path, index=False, compression="snappy")
        except Exception as e:
            raise FlatVectorizeError(f"Fejl ved skriv af Parquet til {pq_path}: {e}")

        # CSV: first 1000 + last 1000 rows
        n = len(out_df)
        if n <= 2000:
            csv_df = out_df
        else:
            head = out_df.head(1000)
            tail = out_df.tail(1000)
            csv_df = pd.concat([head, tail], ignore_index=True)
        csv_df.to_csv(csv_path, index=False)
        print(f"Skrev Parquet {len(out_df)} rækker til {pq_path}")
        print(f"Skrev CSV (sampled) {len(csv_df)} rækker til {csv_path}")
    else:
        if not args.refdate or not args.ticker:
            print("TIP: Uden --all-refdates bør du angive både --ticker og --refdate for et enkelt vindue.\n"
                  "      Alternativt brug --all-refdates for at behandle hele datasættet.", file=sys.stderr)
        # single: produce one-row dataframe then write parquet + sampled csv
        g = select_group(df, args.ticker, args.refdate)
        if g.empty:
            raise SystemExit("Ingen rækker matcher det valgte filter (--ticker/--refdate).")
        X, n = window_from_group(g, args.features, args.window, args.require_consecutive)
        if X is None:
            raise SystemExit(f"Kun {n} rækker i vindue – mindre end ønsket {args.window}. Prøv uden --require-consecutive eller brug kortere --window.")
        Xs = scale_window(X, args.scale)
        v = flatten_row(Xs)
        meta = {
            "Ticker": g["Ticker"].iloc[0],
            "RefDate": g["RefDate"].iloc[0],
            "Window": args.window,
            "Features": "|".join(args.features),
            "Scale": args.scale,
        }
        cols = [f"v{i}" for i in range(len(v))]
        out_df = pd.DataFrame([{**meta, **{c: v[i] for i, c in enumerate(cols)}}])
        # write parquet and csv (csv is full here since single row)
        out_path = args.out
        out_p = Path(out_path)
        if out_p.suffix == '':
            out_p_dir = out_p
            out_p_dir.mkdir(parents=True, exist_ok=True)
            pq_path = out_p_dir / f"{meta['Ticker']}_vector.parquet"
            csv_path = out_p_dir / f"{meta['Ticker']}_vector.csv"
        else:
            pq_path = out_p.with_suffix('.parquet')
            csv_path = out_p.with_suffix('.csv')
            out_p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(pq_path, index=False, compression="snappy")
        out_df.to_csv(csv_path, index=False)
        print(f"Skrev Parquet 1 række til {pq_path}")
        print(f"Skrev CSV 1 række til {csv_path}")

def _main_safe():
    try:
        main()
    except FlatVectorizeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    _main_safe()
