import pandas as pd
import sys
from pathlib import Path
from src import fetch_history_pro as fh


def test_actions_rounding(monkeypatch, tmp_path):
    # Create input ticker CSV
    tickers_csv = tmp_path / "tickers.csv"
    tickers_csv.write_text("ticker\nTST\n", encoding="utf-8")

    # Fake price batch
    def fake_batch(tickers):
        return {"TST": pd.DataFrame({
            "Ticker": ["TST"],
            "Date": [pd.Timestamp("2024-01-01")],
            "Open": [1.111111],
            "High": [1.222222],
            "Low": [1.000001],
            "Close": [1.333333],
            "AdjClose": [1.444444],
            "Volume": [100]
        })}

    monkeypatch.setattr(fh, "batch_download", fake_batch)

    # Fake actions
    def fake_actions(ticker):
        div = pd.DataFrame({"Ticker":[ticker], "Date":[pd.Timestamp("2024-01-01")], "Dividend":[0.1234567]})
        spl = pd.DataFrame({"Ticker":[ticker], "Date":[pd.Timestamp("2024-01-01")], "SplitRatio":[1.2345678]})
        return {"dividends": div, "splits": spl}

    monkeypatch.setattr(fh, "fetch_actions", fake_actions)

    out_dir = tmp_path / "out"
    argv = [
        "src.fetch_history_pro",
        "--input", str(tickers_csv),
        "--out", str(out_dir),
        "--actions",
        "--float-dp", "4",
        "--batch-size", "10"
    ]
    old = sys.argv
    sys.argv = argv
    try:
        fh.main()
    finally:
        sys.argv = old

    div_csv = out_dir / "TST_dividends.csv"
    assert div_csv.exists()
    text = div_csv.read_text(encoding="utf-8").strip().splitlines()[-1]
    # Dividend should be rounded to 4 decimals
    assert text.endswith("0.1235")

    spl_csv = out_dir / "TST_splits.csv"
    assert spl_csv.exists()
    text2 = spl_csv.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert text2.endswith("1.2346")
