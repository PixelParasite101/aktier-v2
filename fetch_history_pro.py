# fetch_history_pro.py
# Robust 1d-historik downloader m. retries, incremental, batch, actions, UTC, logging og rapport.
# Krav: pip install yfinance pandas pyarrow tenacity

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# --- Konstanter ---
DEFAULT_INTERVAL = "1d"  # låst til 1d ifølge krav
REPORT_NAME = "run_report.json"

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Hent fuld historik (1d) for aktier fra CSV med tickers.")
    p.add_argument("--input","-i", required=False, default=None, help="CSV med kolonnen 'ticker'.")
    p.add_argument("--out","-o", default="data", help="Output-mappe.")
    p.add_argument("--per-ticker", action="store_true", help="Gem én fil pr. ticker i stedet for samlet.")
    p.add_argument("--batch-size", type=int, default=30, help="Antal tickers pr. batch (yfinance multi-download).")
    p.add_argument("--incremental", action="store_true", help="Hent kun nye datoer (byg på eksisterende filer).")
    p.add_argument("--actions", action="store_true", help="Gem udbytter/splits pr. ticker.")
    p.add_argument("--adjusted-only", action="store_true", help="Gem kun AdjClose som Close.")
    p.add_argument("--compression", default="snappy", help="Parquet-kompression: snappy|zstd|gzip.")
    p.add_argument("--partition-by", default=None, help="Parquet-partitionering, fx 'Ticker'. (kun samlet tilstand)")
    p.add_argument("--only", default=None, help="Kun disse tickers (kommasepareret).")
    p.add_argument("--validate-only", action="store_true", help="Valider input og net—ingen downloads.")
    p.add_argument("--dry-run", action="store_true", help="Download i RAM, men skriv ikke til disk.")
    p.add_argument("--fail-on-empty", type=int, default=None, help="Exit 1 hvis tomme tickers > N.")
    p.add_argument("--log-file", default=None, help="Fil til JSON-lines logs (append).")
    p.add_argument(
        "--float-dp",
        type=int,
        default=None,
        help="Antal decimaler for floats i CSV (Parquet bevarer fuld præcision).",
    )
    p.add_argument(
        "--preset",
        choices=["standard", "fast", "validate"],
        default=None,
        help="Forudindstillet kørsel: standard|fast|validate. Manuelle flags kan stadig override.",
    )
    return p.parse_args()

# --- Preset helpers ---
def _flag_provided(*names: str) -> bool:
    return any(n in sys.argv for n in names)

def apply_preset(args):
    if not args.preset:
        return args
    preset = args.preset
    if preset == "standard":
        if args.input is None and not _flag_provided("--input", "-i"):
            args.input = "aktie.csv"
        if (args.out == "data" and not _flag_provided("--out", "-o")) or args.out is None:
            args.out = "data_all"
        if not _flag_provided("--incremental"):
            args.incremental = True
        if not _flag_provided("--actions"):
            args.actions = True
        if args.partition_by in (None, "") and not _flag_provided("--partition-by"):
            args.partition_by = "Ticker"
        if args.batch_size == 30 and not _flag_provided("--batch-size"):
            args.batch_size = 30
        if args.compression == "snappy" and not _flag_provided("--compression"):
            args.compression = "snappy"
        if args.float_dp is None and not _flag_provided("--float-dp"):
            args.float_dp = 4
        # keep adjusted_only default (False) unless user sets it
        # per-ticker remains default False unless user sets it
    elif preset == "fast":
        if args.input is None and not _flag_provided("--input", "-i"):
            args.input = "aktie.csv"
        if (args.out == "data" and not _flag_provided("--out", "-o")) or args.out is None:
            args.out = "data_all"
        if not _flag_provided("--incremental"):
            args.incremental = True
        # fast: skip actions by default
        # only set actions if user explicitly asked
        if args.partition_by in (None, "") and not _flag_provided("--partition-by"):
            args.partition_by = "Ticker"
        if not _flag_provided("--batch-size"):
            args.batch_size = 20
        if not _flag_provided("--compression"):
            args.compression = "snappy"
        if args.float_dp is None and not _flag_provided("--float-dp"):
            args.float_dp = 4
    elif preset == "validate":
        if args.input is None and not _flag_provided("--input", "-i"):
            args.input = "aktie.csv"
        if (args.out == "data" and not _flag_provided("--out", "-o")) or args.out is None:
            args.out = "data_all"
        if not _flag_provided("--validate-only"):
            args.validate_only = True
    return args

# --- Utility: logging ---
def log_event(log_file: Optional[str], event: Dict):
    if not log_file:
        return
    event = dict(event)
    event["ts"] = datetime.utcnow().isoformat()+"Z"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# --- I/O helpers ---
def ensure_out(path: str):
    os.makedirs(path, exist_ok=True)

def read_table(path: str, only: Optional[str]) -> List[str]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("CSV skal have kolonnen 'ticker'.")
    tickers = df["ticker"].astype(str).str.strip()
    tickers = [t for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))  # unikke, stable order
    if only:
        wanted = {t.strip() for t in only.split(",") if t.strip()}
        tickers = [t for t in tickers if t in wanted]
        missing = wanted - set(tickers)
        if missing:
            sys.stderr.write(f"Advarsel: disse 'only' fandtes ikke i input: {sorted(missing)}\n")
    if not tickers:
        raise ValueError("Ingen gyldige tickers i CSV.")
    return tickers

def read_existing_parquet(out_dir: str, per_ticker: bool) -> Optional[pd.DataFrame]:
    if per_ticker:
        return None
    p = os.path.join(out_dir, "history_all.parquet")
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return None

def read_existing_ticker_parquet(out_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    p = os.path.join(out_dir, f"{ticker}.parquet")
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None

# --- Normalisering / merge ---
def normalize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker","Date","Open","High","Low","Close","AdjClose","Volume"])
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df.rename(columns={"Adj Close":"AdjClose", date_col:"Date"}, inplace=True)
    # UTC
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        if getattr(df["Date"].dt, "tz", None) is not None:
            df["Date"] = df["Date"].dt.tz_convert(timezone.utc).dt.tz_localize(None)
    # Sikr kolonner
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["Ticker"] = ticker
    # Typer
    for c in ["Open","High","Low","Close","AdjClose"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("Int64")
    df = df[["Ticker","Date","Open","High","Low","Close","AdjClose","Volume"]]
    df = df.drop_duplicates(subset=["Ticker","Date"]).sort_values(["Ticker","Date"])
    return df

def incremental_merge(existing: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df
    all_df = pd.concat([existing, new_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["Ticker","Date"]).sort_values(["Ticker","Date"])
    return all_df

# --- Actions (udbytter/splits) ---
def fetch_actions(ticker: str) -> Dict[str, pd.DataFrame]:
    t = yf.Ticker(ticker)
    div = t.dividends.reset_index()
    spl = t.splits.reset_index()
    if not div.empty:
        div.rename(columns={"Date":"Date","Dividends":"Dividend"}, inplace=True)
        if pd.api.types.is_datetime64_any_dtype(div["Date"]) and getattr(div["Date"].dt, "tz", None) is not None:
            div["Date"] = div["Date"].dt.tz_convert(timezone.utc).dt.tz_localize(None)
        div["Ticker"] = ticker
        div = div[["Ticker","Date","Dividend"]]
    else:
        div = pd.DataFrame(columns=["Ticker","Date","Dividend"])
    if not spl.empty:
        spl.rename(columns={"Date":"Date","Stock Splits":"SplitRatio"}, inplace=True)
        if pd.api.types.is_datetime64_any_dtype(spl["Date"]) and getattr(spl["Date"].dt, "tz", None) is not None:
            spl["Date"] = spl["Date"].dt.tz_convert(timezone.utc).dt.tz_localize(None)
        spl["Ticker"] = ticker
        spl = spl[["Ticker","Date","SplitRatio"]]
    else:
        spl = pd.DataFrame(columns=["Ticker","Date","SplitRatio"])
    return {"dividends": div, "splits": spl}

# --- Download (batch) ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=8))
def batch_download(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    # group_by='ticker' returnerer kolonner per ticker; period='max' for fuld historik (daglig)
    data = yf.download(tickers=tickers, period="max", interval=DEFAULT_INTERVAL,
                       progress=False, auto_adjust=False, threads=False, group_by="ticker")
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = data[t]
        except Exception:
            # single-frame fallback hvis kun 1 ticker
            sub = data if isinstance(data, pd.DataFrame) else None
        out[t] = normalize_prices(sub, t) if isinstance(sub, pd.DataFrame) else normalize_prices(pd.DataFrame(), t)
    return out

# --- Gemning ---
def save_prices(df: pd.DataFrame, out_dir: str, per_ticker: bool, compression: str, partition_by: Optional[str], float_dp: Optional[int]):
    # Afrunding af numeriske prisfelter hvis ønsket (KUN til CSV)
    csv_float_format = None
    df_csv = df
    if float_dp is not None:
        csv_float_format = f"%.{float_dp}f"
        cols_to_round = [c for c in ["Open","High","Low","Close","AdjClose"] if c in df.columns]
        if cols_to_round:
            df_csv = df.copy()
            for c in cols_to_round:
                df_csv[c] = pd.to_numeric(df_csv[c], errors="coerce").round(float_dp)
    if per_ticker:
        # forventer kun én ticker pr. df
        tick = df_csv["Ticker"].iloc[0]
        df_csv.to_csv(os.path.join(out_dir, f"{tick}.csv"), index=False, float_format=csv_float_format)
        try:
            # Parquet gemmes i fuld præcision
            df.to_parquet(os.path.join(out_dir, f"{tick}.parquet"), index=False, compression=compression)
        except Exception as e:
            sys.stderr.write(f"Parquet-fejl for {tick}: {e}\n")
        return
    # samlet
    csv_path = os.path.join(out_dir, "history_all.csv")
    df_csv.to_csv(csv_path, index=False, float_format=csv_float_format)
    try:
        if partition_by:
            pq_dir = os.path.join(out_dir, "history_all_parquet")  # mappe ved partitionering
            # Parquet gemmes i fuld præcision
            df.to_parquet(pq_dir, index=False, compression=compression, partition_cols=[partition_by])
        else:
            pq_path = os.path.join(out_dir, "history_all.parquet")
            # Parquet gemmes i fuld præcision
            df.to_parquet(pq_path, index=False, compression=compression)
    except Exception as e:
        sys.stderr.write(f"Parquet-fejl (samlet): {e}\n")

# --- Main ---
def main():
    args = parse_args()
    # If run without any CLI args (e.g., VS Code "Run Python File"), default to preset=standard
    if args.preset is None and len(sys.argv) == 1:
        args.preset = "standard"
    args = apply_preset(args)
    if not args.input:
        raise SystemExit("Fejl: --input mangler. Angiv --input eller brug --preset standard/fast/validate.")
    ensure_out(args.out)

    # Læs tickers
    tickers = read_table(args.input, args.only)
    if args.validate_only:
        print(f"OK: {len(tickers)} tickers fundet. Validate-only, ingen download.")
        return

    report = {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "interval": DEFAULT_INTERVAL,
        "ok": [],
        "failed": [],
        "summary": {}
    }

    combined = [] if not args.per_ticker else None
    existing_all = read_existing_parquet(args.out, args.per_ticker) if args.incremental else None

    # Batch loop
    n = len(tickers)
    batches = [tickers[i:i+args.batch_size] for i in range(0, n, args.batch_size)]

    for group in batches:
        t0 = time.time()
        try:
            result = batch_download(group)
            log_event(args.log_file, {"event":"batch_ok", "size": len(group), "elapsed_s": round(time.time()-t0,3)})
        except Exception as e:
            log_event(args.log_file, {"event":"batch_fail", "size": len(group), "error": str(e)})
            # marker alle som failed i denne batch
            for t in group:
                report["failed"].append({"ticker": t, "reason": str(e)})
            time.sleep(1.0)
            continue

        for t in group:
            df = result.get(t, pd.DataFrame())
            if df.empty:
                report["failed"].append({"ticker": t, "reason": "empty"})
                log_event(args.log_file, {"event":"ticker_empty", "ticker": t})
                continue

            if args.adjusted_only:
                df["Close"] = df["AdjClose"]

            # incremental
            if args.incremental:
                if args.per_ticker:
                    old = read_existing_ticker_parquet(args.out, t)
                    df = incremental_merge(old, df)
                else:
                    # samlet merge udføres først til sidst — her bare saml
                    pass

            # write?
            if args.per_ticker:
                if not args.dry_run:
                    save_prices(df, args.out, per_ticker=True, compression=args.compression, partition_by=None, float_dp=args.float_dp)
            else:
                combined.append(df)

            # actions
            if args.actions:
                try:
                    acts = fetch_actions(t)
                    if not args.dry_run:
                        # CSV: afrund hvis ønsket; Parquet: fuld præcision
                        div_csv_df = acts["dividends"]
                        spl_csv_df = acts["splits"]
                        csv_float_format = None
                        if args.float_dp is not None:
                            csv_float_format = f"%.{args.float_dp}f"
                            if not div_csv_df.empty and "Dividend" in div_csv_df.columns:
                                div_csv_df = div_csv_df.copy()
                                div_csv_df["Dividend"] = pd.to_numeric(div_csv_df["Dividend"], errors="coerce").round(args.float_dp)
                            if not spl_csv_df.empty and "SplitRatio" in spl_csv_df.columns:
                                spl_csv_df = spl_csv_df.copy()
                                spl_csv_df["SplitRatio"] = pd.to_numeric(spl_csv_df["SplitRatio"], errors="coerce").round(args.float_dp)
                        if not div_csv_df.empty:
                            div_csv_df.to_csv(
                                os.path.join(args.out, f"{t}_dividends.csv"),
                                index=False,
                                float_format=csv_float_format,
                            )
                            try:
                                acts["dividends"].to_parquet(
                                    os.path.join(args.out, f"{t}_dividends.parquet"),
                                    index=False,
                                    compression=args.compression,
                                )
                            except Exception:
                                pass
                        if not spl_csv_df.empty:
                            spl_csv_df.to_csv(
                                os.path.join(args.out, f"{t}_splits.csv"),
                                index=False,
                                float_format=csv_float_format,
                            )
                            try:
                                acts["splits"].to_parquet(
                                    os.path.join(args.out, f"{t}_splits.parquet"),
                                    index=False,
                                    compression=args.compression,
                                )
                            except Exception:
                                pass
                except Exception as e:
                    report["failed"].append({"ticker": t, "reason": f"actions:{e}"})
                    log_event(args.log_file, {"event":"actions_fail", "ticker": t, "error": str(e)})

            report["ok"].append({"ticker": t, "rows": int(len(df))})
            log_event(args.log_file, {"event":"ticker_ok", "ticker": t, "rows": int(len(df))})

        time.sleep(0.5)  # venlig mod Yahoo

    # Samlet gemning
    if not args.per_ticker and combined:
        all_df = pd.concat(combined, ignore_index=True)
        if args.incremental and existing_all is not None:
            all_df = incremental_merge(existing_all, all_df)
        if not args.dry_run:
            save_prices(
                all_df,
                args.out,
                per_ticker=False,
                compression=args.compression,
                partition_by=args.partition_by,
                float_dp=args.float_dp,
            )

    # Rapport
    empty_n = sum(1 for f in report["failed"] if f.get("reason") == "empty")
    report["summary"] = {
        "tickers_total": n,
        "tickers_ok": len(report["ok"]),
        "tickers_failed": len(report["failed"]),
        "empty_series": empty_n,
        "out_dir": args.out,
        "per_ticker": bool(args.per_ticker),
        "incremental": bool(args.incremental),
        "actions": bool(args.actions),
        "adjusted_only": bool(args.adjusted_only),
        "dry_run": bool(args.dry_run)
    }
    if not args.dry_run:
        with open(os.path.join(args.out, REPORT_NAME), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Færdig. OK: {len(report['ok'])}  Fejl: {len(report['failed'])}  Output: {args.out}")
    if args.fail_on_empty is not None and empty_n > args.fail_on_empty:
        sys.exit(1)

if __name__ == "__main__":
    main()
