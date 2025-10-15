import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Missing dependency: install yfinance (pip install yfinance) and rerun.")
    raise

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    print("Missing dependency: install scikit-learn (pip install scikit-learn) and rerun.")
    raise

# Settings
ROOT = Path(__file__).resolve().parents[1]
WATCH_CSV = ROOT / "watch.csv"
FLAT_VECTORS_FILE = ROOT / "data" / "flat_vectors" / "all_vectors.parquet"
OUT_CSV = ROOT / "data" / "flat_vectors" / "watch_matches.csv"  # top K (standard)
OUT_CSV_FULL = ROOT / "data" / "flat_vectors" / "watch_matches_full.csv"  # fuld rangliste
OUT_QUERY_VECTOR = ROOT / "data" / "flat_vectors" / "watch_query_vector.csv"  # den anvendte (auto-trimmede) query-vektor
WINDOW_BEFORE = 20  # standard fallback hvis vi ikke auto-tilpasser

def compute_indicators(df):
    # df must have ['Open','High','Low','Close','Volume'] and index = Date
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    features = pd.DataFrame(index=df.index)
    features["close"] = close
    features["ret_1"] = close.pct_change(1)
    features["sma_5"] = close.rolling(5, min_periods=1).mean()
    features["sma_10"] = close.rolling(10, min_periods=1).mean()
    features["ema_12"] = close.ewm(span=12, adjust=False).mean()
    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    features["rsi_14"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    # ATR 14
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    # Volume features
    features["vol"] = vol
    features["vol_roc_5"] = vol.pct_change(5)
    # Fill or keep NaN? we will drop rows with NaN later
    return features

def flatten_window(mat):
    # mat shape (window, features) -> flattened row-major
    return mat.ravel(order="C")


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def build_rebased_from_history(df: pd.DataFrame, ticker: str, before: int = 20, after: int = 5):
    """Build rebased rows for each RefDate where full window exists.

    Returns a DataFrame with columns matching files in data/rebased/*.csv
    """
    df2 = df.copy()
    # ensure Date index is datetime
    if not isinstance(df2.index, pd.DatetimeIndex):
        try:
            df2.index = pd.to_datetime(df2.index)
        except Exception:
            pass

    # use AdjClose as Close here (no adjustments available at download time)
    df2['AdjClose'] = df2['Close']
    # Moving averages
    df2['MA_20'] = df2['AdjClose'].rolling(20, min_periods=1).mean()
    df2['MA_50'] = df2['AdjClose'].rolling(50, min_periods=1).mean()
    df2['MA_200'] = df2['AdjClose'].rolling(200, min_periods=1).mean()
    # RSI
    df2['RSI_14'] = rsi_wilder(df2['AdjClose'], 14)

    rows = []
    n = len(df2)
    # indices where full window exists: i in [before, n-after-1]
    for ref_idx in range(before, n - after):
        ref_date = df2.index[ref_idx]
        base_vals = df2.iloc[ref_idx]
        # base for rebasing: price-like columns
        base_adj = float(base_vals['AdjClose']) if not pd.isna(base_vals['AdjClose']) else None
        base_close = float(base_vals['Close']) if not pd.isna(base_vals['Close']) else None
        base_ma20 = float(base_vals['MA_20']) if not pd.isna(base_vals['MA_20']) else None
        base_ma50 = float(base_vals['MA_50']) if not pd.isna(base_vals['MA_50']) else None
        base_ma200 = float(base_vals['MA_200']) if not pd.isna(base_vals['MA_200']) else None

        for offset, idx in enumerate(range(ref_idx - before, ref_idx + after + 1), start=-before):
            row_vals = df2.iloc[idx]
            date = df2.index[idx]
            # compute rebased price-like columns: value / base * 100
            def rb(val, base):
                try:
                    if base is None or base == 0 or pd.isna(val):
                        return float('nan')
                    return float(val) / float(base) * 100.0
                except Exception:
                    return float('nan')

            AdjClose_Rebased = rb(row_vals['AdjClose'], base_adj)
            Close_Rebased = rb(row_vals['Close'], base_close)
            MA_20_Rebased = rb(row_vals['MA_20'], base_ma20)
            MA_50_Rebased = rb(row_vals['MA_50'], base_ma50)
            MA_200_Rebased = rb(row_vals['MA_200'], base_ma200)
            RSI_14 = float(row_vals['RSI_14']) if not pd.isna(row_vals['RSI_14']) else float('nan')

            rows.append({
                'Ticker': ticker,
                'RefDate': ref_date.strftime('%Y-%m-%d'),
                'Offset': offset,
                'Date': date.strftime('%Y-%m-%d'),
                'AdjClose_Rebased': AdjClose_Rebased,
                'Close_Rebased': Close_Rebased,
                'MA_20_Rebased': MA_20_Rebased,
                'MA_50_Rebased': MA_50_Rebased,
                'MA_200_Rebased': MA_200_Rebased,
                'RSI_14': RSI_14,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def load_flat_vectors(path):
    if not path.exists():
        raise FileNotFoundError(f"Flat vectors file not found: {path}")
    fv = pd.read_parquet(path)
    return fv

def detect_vector_columns(fv, expected_len):
    """Heuristisk detektion af flad-vektor kolonner.

    expected_len: længden af query-vektoren (window * features).
    Returnerer en liste af kolonnenavne i samme rækkefølge som de skal bruges.
    """
    # Udvidet exclude med almindelige meta/konfig kolonner
    exclude = {"Ticker", "RefDate", "Date", "index", "Window", "Features", "Scale"}
    # Behold kun numeriske ikke-meta kolonner (bevar kolonne-orden)
    candidate_cols = [c for c in fv.columns if c not in exclude and pd.api.types.is_numeric_dtype(fv[c])]
    # Hvis vi har eksplicit v0..vN kolonner, foretræk dem.
    import re
    v_pattern = re.compile(r"^v\d+$")
    prefixed = [c for c in fv.columns if c not in exclude and pd.api.types.is_numeric_dtype(fv[c]) and (c.startswith("f_") or c.startswith("feat_") or v_pattern.match(c))]
    # Hvis vi har mindst det antal vi behøver i prefixed-listen, trim og returner (bevarer rækkefølge)
    if len(prefixed) >= expected_len:
        if len(prefixed) > expected_len:
            print(f"ADVARSEL: Flat vector datasæt har {len(prefixed)} matchende 'prefixed' kolonner, men forventede {expected_len}. Trimmer til de første {expected_len}.")
        return prefixed[:expected_len]
    # Direkte match på længde for alle numeriske kandidater
    if len(candidate_cols) == expected_len:
        return candidate_cols
    # Hvis vi har flere end expected_len kan vi trimme numeriske kandidater som en sidste udvej
    if len(candidate_cols) > expected_len:
        # Trim og advare
        print(f"ADVARSEL: Flat vector datasæt har {len(candidate_cols)} numeriske kandidater, men forventede {expected_len}. Trimmer til de første {expected_len}.")
        return candidate_cols[:expected_len]
    # Fejl – byg detaljeret besked
    msg_parts = [
        f"Kunne ikke matche længde. Query-vektor længde={expected_len}.",
        f"Antal numeriske kandidater (ekskl. meta)={len(candidate_cols)}.",
    ]
    # Forsøg at hjælpe: Hvis datasættet har vindue/feature metadata kan vi estimere forventet længde derfra
    if {"Window", "Features"}.issubset(fv.columns):
        try:
            window_vals = fv["Window"].dropna().unique()
            features_vals = fv["Features"].dropna().unique()
            if len(window_vals) == 1 and len(features_vals) == 1:
                w = int(window_vals[0])
                f = len(str(features_vals[0]).split("|"))
                ds_len = w * f
                msg_parts.append(f"Dataset ser ud til at bruge Window={w} og {f} features => {ds_len} dimensioner.")
                msg_parts.append("Din query bruger et andet (window * features). Sørg for at scripts bruger samme feature-liste og vindueslængde.")
        except Exception:
            pass
    msg_parts.append("Løsninger: (1) Tilpas WINDOW_BEFORE og featurelisten til dataset. (2) Regenerér flat vectors med dine indikatorer. (3) Implementér en kompatibilitets-tilstand.")
    raise ValueError(" ".join(msg_parts))

def main():
    if not WATCH_CSV.exists():
        print("watch.csv not found at:", WATCH_CSV)
        sys.exit(1)
    ticker = pd.read_csv(WATCH_CSV).iloc[0, 0].strip()
    print("Ticker from watch:", ticker)
    # prepare per-ticker output directory early so we can save intermediate files
    per_ticker_dir = ROOT / 'data' / 'watch_flat_vectors' / ticker.replace('/', '_')
    per_ticker_dir.mkdir(parents=True, exist_ok=True)

    # Load flat vectors early so we can detect the dataset's window/features
    try:
        fv = load_flat_vectors(FLAT_VECTORS_FILE)
        # detect target window/features if present
        target_window = None
        target_n_features = None
        if {"Window", "Features"}.issubset(fv.columns):
            try:
                w_vals = fv["Window"].dropna().unique()
                f_vals = fv["Features"].dropna().unique()
                if len(w_vals) == 1 and len(f_vals) == 1:
                    target_window = int(w_vals[0])
                    target_n_features = len(str(f_vals[0]).split("|"))
            except Exception:
                target_window = None
                target_n_features = None
    except Exception:
        fv = None
        target_window = None
        target_n_features = None

    # fetch daily OHLCV, retrying with larger periods if computed indicators have too few rows
    periods_to_try = ["1y", "2y", "5y", "max"]
    df = None
    computed_features = None
    required_rows = WINDOW_BEFORE
    if target_window:
        required_rows = max(required_rows, int(target_window))
    for period in periods_to_try:
        print(f"Downloading history (period={period})...")
        df_try = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if df_try.empty:
            continue
        df_try = df_try[["Open", "High", "Low", "Close", "Volume"]].dropna()
        # compute indicators and count non-NaN rows
        feats_try = compute_indicators(df_try)
        feats_try = feats_try.dropna()
        if feats_try.shape[0] >= required_rows:
            df = df_try
            computed_features = feats_try
            print(f"Got {feats_try.shape[0]} non-NaN indicator rows with period={period}")
            break
        # otherwise keep last successful as fallback but continue trying
        df = df_try
        computed_features = feats_try

    if df is None or df.empty:
        print("No historical data returned for", ticker)
        sys.exit(1)
    if df.empty:
        print("No historical data returned for", ticker)
        sys.exit(1)
    # save raw historical OHLCV to CSV (include Date column)
    try:
        hist_csv = per_ticker_dir / 'history_raw.csv'
        df_reset = df.reset_index()
        df_reset.to_csv(hist_csv, index=False, float_format='%.4f')
        print('Wrote historical CSV to', hist_csv)
    except Exception as e:
        print('WARN: could not write historical CSV:', e)
    # create a simple 'rebased' version where Close (and prices) are normalized so first Close == 100
    try:
        rebased_csv = per_ticker_dir / 'history_rebased.csv'
        df_rb = df.reset_index().copy()
        if not df_rb.empty:
            base_close = float(df_rb['Close'].iloc[0])
            if base_close != 0:
                for c in ['Open', 'High', 'Low', 'Close']:
                    df_rb[f'{c}_rebased'] = df_rb[c] / base_close * 100.0
        df_rb.to_csv(rebased_csv, index=False, float_format='%.4f')
        print('Wrote rebased CSV to', rebased_csv)
    except Exception as e:
        print('WARN: could not write rebased CSV:', e)

    # Build rebased windows in the same format as data/rebased/* and write to per_ticker_dir
    try:
        reb_df = build_rebased_from_history(df, ticker, before=20, after=5)
        if not reb_df.empty:
            reb_out = per_ticker_dir / f"{ticker.replace('/','_')}_rebased.csv"
            reb_df.to_csv(reb_out, index=False, float_format='%.4f')
            print('Wrote rebased windows to', reb_out)
    except Exception as e:
        print('WARN: could not build/write rebased windows:', e)

    # use computed features from the successful download attempt
    if computed_features is not None:
        features = computed_features
    else:
        features = compute_indicators(df)
        features = features.dropna()

    if features.shape[0] < 1:
        print("Not enough non-NaN indicator rows after indicator computation:", features.shape[0])
        sys.exit(1)

    # take last WINDOW_BEFORE days (initial query basis)
    window_df = features.iloc[-WINDOW_BEFORE:, :].copy()
    Xq_full = window_df.to_numpy(dtype=float)  # shape (W0, F0)
    base_window = Xq_full.shape[0]
    base_feats = Xq_full.shape[1]
    q_vec = flatten_window(Xq_full)

    # load existing flat vectors
    fv = load_flat_vectors(FLAT_VECTORS_FILE)
    # Auto-tilpas hvis dataset har konsistent Window & Features metadata der afviger fra vores
    target_window = None
    target_n_features = None
    if {"Window", "Features"}.issubset(fv.columns):
        try:
            w_vals = fv["Window"].dropna().unique()
            f_vals = fv["Features"].dropna().unique()
            if len(w_vals) == 1 and len(f_vals) == 1:
                target_window = int(w_vals[0])
                target_n_features = len(str(f_vals[0]).split("|"))
        except Exception:
            target_window = None
            target_n_features = None

    if target_window and target_n_features:
        target_len = target_window * target_n_features
        current_len = q_vec.size
        if target_len != current_len:
            if target_window > base_window:
                print(f"ADVARSEL: Dataset kræver window={target_window}, men vi har kun {base_window} dage. Kan ikke udvide – forsøger stadig med vores kortere vindue.")
            else:
                # trim vindue
                if target_window < base_window:
                    window_df = window_df.iloc[-target_window:, :]
                    Xq_full = window_df.to_numpy(dtype=float)
                    base_window = target_window
            if target_n_features < base_feats:
                # trim features til de første N (beholder kolonneordenen vi lavede)
                kept_cols = window_df.columns[:target_n_features]
                window_df = window_df[kept_cols]
                Xq_full = window_df.to_numpy(dtype=float)
                base_feats = target_n_features
            elif target_n_features > base_feats:
                print(f"ADVARSEL: Dataset bruger {target_n_features} features men vi genererede kun {base_feats}. Kan ikke tilføje manglende – fortsætter med {base_feats}.")
            q_vec = flatten_window(Xq_full)
            print(f"Auto-tilpasning: query dimension ændret til window={base_window}, features={base_feats} (len={q_vec.size}).")

    # detect vector columns baseret på (evt. auto-justeret) q_vec
    expected_len = q_vec.size
    try:
        vec_cols = detect_vector_columns(fv, expected_len)
    except Exception as e:
        print("Error detecting vector columns:", e)
        sys.exit(1)

    print("Using %d vector columns for comparison" % len(vec_cols))
    X = fv[vec_cols].to_numpy(dtype=float)
    # normalize vectors to unit length to avoid zero-division
    # use cosine similarity (sklearn)
    sims = cosine_similarity(q_vec.reshape(1, -1), X).ravel()

    # Eksportér den anvendte query-vektor (så vi kan inspicere hvad der blev matchet imod)
    try:
        OUT_QUERY_VECTOR.parent.mkdir(parents=True, exist_ok=True)
        q_meta = {
            "Ticker": ticker,
            "WindowUsed": base_window,
            "FeaturesUsed": base_feats,
            "VectorLength": int(q_vec.size),
            "OriginalWindowRequested": WINDOW_BEFORE,
        }
        # v0..v{n-1}
        q_cols = {f"v{i}": float(q_vec[i]) for i in range(q_vec.size)}
        pd.DataFrame([{**q_meta, **q_cols}]).to_csv(OUT_QUERY_VECTOR, index=False, float_format='%.4f')
        print("Query vector written to", OUT_QUERY_VECTOR)
    except Exception as e:
        print("WARN: Could not write query vector CSV:", e)
    fv = fv.assign(sim_to_watch=sims)
    ranked = fv.sort_values("sim_to_watch", ascending=False)
    top = ranked.head(20)
    print(top[["Ticker", "RefDate", "sim_to_watch"]].to_string(index=False))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Gem i per-ticker outputmappe: data/watch_flat_vectors/<TICKER>/
    out_dir = ROOT / 'data' / 'watch_flat_vectors' / ticker.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_top = out_dir / 'watch_matches.csv'
    out_full = out_dir / 'watch_matches_full.csv'
    out_query = out_dir / 'watch_query_vector.csv'

    # skriv top 20
    top.to_csv(out_top, index=False, float_format='%.4f')
    # skriv fuld liste
    ranked.to_csv(out_full, index=False, float_format='%.4f')
    # skriv query vektor (overskriv tidligere)
    try:
        q_meta = {
            "Ticker": ticker,
            "WindowUsed": base_window,
            "FeaturesUsed": base_feats,
            "VectorLength": int(q_vec.size),
            "OriginalWindowRequested": WINDOW_BEFORE,
        }
        q_cols = {f"v{i}": float(q_vec[i]) for i in range(q_vec.size)}
        pd.DataFrame([{**q_meta, **q_cols}]).to_csv(out_query, index=False, float_format='%.4f')
    except Exception:
        pass
    # Skriv en schema/header CSV (kun kolonner) så andre værktøjer kan følge samme kolonne-layout
    try:
        schema_path = out_dir / 'watch_matches_schema.csv'
        pd.DataFrame(columns=ranked.columns).to_csv(schema_path, index=False, float_format='%.4f')
    except Exception:
        pass

    print("Top matches written to", out_top)
    print("Full ranked matches written to", out_full)
    print("Query vector written to", out_query)

if __name__ == "__main__":
    main()