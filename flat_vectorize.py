#!/usr/bin/env python3
"""
flat_vectorize.py
Laver flade vektorer af tidsvinduer fra en rebased CSV med kolonner som:
Ticker, RefDate, Offset, Date, AdjClose_Rebased, Close_Rebased,
MA_20_Rebased, MA_50_Rebased, MA_200_Rebased, RSI_14, ...

Eksempler:
1) Ét vindue (100 dage) for en given RefDate for MSFT med 6 features:
   python flat_vectorize.py --csv MSFT_rebased.csv --ticker MSFT --refdate 1987-04-13 \
       --window 100 --features AdjClose_Rebased Close_Rebased MA_20_Rebased MA_50_Rebased MA_200_Rebased RSI_14 \
       --out vectors_single.csv

2) Vinduer for alle (Ticker, RefDate) (standard 75 dage) med indikatorer (4 features):
   python flat_vectorize.py --csv MSFT_rebased.csv \
       --window 75 --features MA_20_Rebased MA_50_Rebased MA_200_Rebased RSI_14 \
       --all-refdates --out vectors_all.csv

3) Samme som (2) men z-score normaliseret per vindue:
   python flat_vectorize.py --csv MSFT_rebased.csv --window 75 \
       --features MA_20_Rebased MA_50_Rebased MA_200_Rebased RSI_14 \
       --all-refdates --scale zscore --out vectors_all_z.csv
"""
import argparse
import sys
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Sti til input CSV")
    p.add_argument("--out", required=True, help="Sti til output CSV")
    p.add_argument("--features", nargs="+", required=True,
                   help="Kolonner der bruges som features (rækkefølge = flad rækkefølge pr. dag)")
    p.add_argument("--window", type=int, default=75, help="Antal dage i vinduet (default: 75)")
    p.add_argument("--ticker", help="Filtrér til én Ticker (valgfri)")
    p.add_argument("--refdate", help="Vælg én RefDate (YYYY-MM-DD) for single-vindue")
    p.add_argument("--all-refdates", action="store_true",
                   help="Generér vinduer for alle (Ticker, RefDate) grupper")
    p.add_argument("--require-consecutive", action="store_true",
                   help="Kræv at der findes præcis 'window' på hinanden følgende offsets fra min offset")
    p.add_argument("--scale", choices=["none", "zscore", "minmax"], default="none",
                   help="Skalering pr. vindue: none | zscore | minmax (default: none)")
    return p.parse_args()

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
    """
    Antager at 'g' indeholder rækker for én (Ticker, RefDate), sorteret stigende på Offset.
    Tager de første 'window' rækker (eller præcis en sammenhængende blok hvis require_consecutive).
    Returnerer (matrix shape [window, F], faktisk_window_størrelse)
    """
    g = g.sort_values(["Offset", "Date"], ascending=[True, True])
    # find start så vi har mindst 'window' rækker
    if require_consecutive:
        # check for sekvens af offsets med længde 'window'
        offsets = g["Offset"].to_numpy()
        # normaliser til 0..N-1 ved at trække min
        offs0 = offsets - offsets.min()
        # vi kræver at offs0[0:window] == range(window)
        # ellers prøver vi at glide
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
    # Række-major: dag 1's features, dag 2's features, ...
    return X.flatten()

def process_single(df, args):
    g = select_group(df, args.ticker, args.refdate)
    if g.empty:
        sys.exit("Ingen rækker matcher det valgte filter (--ticker/--refdate).")
    X, n = window_from_group(g, args.features, args.window, args.require_consecutive)
    if X is None:
        sys.exit(f"Kun {n} rækker i vindue – mindre end ønsket {args.window}. Prøv uden --require-consecutive eller brug kortere --window.")
    X = scale_window(X, args.scale)
    v = flatten_row(X)
    meta = {
        "Ticker": g["Ticker"].iloc[0],
        "RefDate": g["RefDate"].iloc[0],
        "Window": args.window,
        "Features": "|".join(args.features),
        "Scale": args.scale,
    }
    # kolonnenavne v0..v{N-1}
    cols = [f"v{i}" for i in range(len(v))]
    out = pd.DataFrame([{**meta, **{c: v[i] for i, c in enumerate(cols)}}])
    out.to_csv(args.out, index=False)
    print(f"Skrev 1 vektor ({len(v)} dim) til {args.out}")

def process_all(df, args):
    rows = []
    group_cols = ["Ticker", "RefDate"]
    for (tic, rd), g in df.groupby(group_cols):
        X, n = window_from_group(g, args.features, args.window, args.require_consecutive)
        if X is None:
            # spring over grupper der ikke har nok data
            continue
        Xs = scale_window(X, args.scale)
        v = flatten_row(Xs)
        row = {
            "Ticker": tic,
            "RefDate": rd,
            "Window": args.window,
            "Features": "|".join(args.features),
            "Scale": args.scale,
        }
        for i, val in enumerate(v):
            row[f"v{i}"] = val
        rows.append(row)

    if not rows:
        sys.exit("Ingen grupper havde nok data til at danne et vindue.")
    out = pd.DataFrame(rows).sort_values(group_cols)
    out.to_csv(args.out, index=False)
    print(f"Skrev {len(out)} flade vektorer (hver på {args.window*len(args.features)} dim) til {args.out}")

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    check_columns(df, ["Ticker", "RefDate", "Offset", "Date"])
    check_columns(df, args.features)

    if args.all_refdates:
        process_all(df, args)
    else:
        if not args.refdate or not args.ticker:
            print("TIP: Uden --all-refdates bør du angive både --ticker og --refdate for et enkelt vindue.\n"
                  "      Alternativt brug --all-refdates for at behandle hele datasættet.", file=sys.stderr)
        process_single(df, args)

if __name__ == "__main__":
    main()
