import json
import sys
from pathlib import Path
import pandas as pd
from src import fetch_history_pro as fh
from src import compute_features as cf
from src import make_rebased_windows as mr


def _run_script(module, argv):
    old = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old


def test_metadata_fetch_features_rebased(tmp_path):
    # Build minimal raw fetch output using monkeypatched network
    tickers_csv = tmp_path / "tickers.csv"
    tickers_csv.write_text("ticker\nAAA\n", encoding="utf-8")

    import pandas as pd

    def fake_batch(tickers):
        return {"AAA": pd.DataFrame({
            "Ticker": ["AAA"],
            "Date": [pd.Timestamp("2024-01-01")],
            "Open": [1.0], "High": [1.1], "Low": [0.9],
            "Close": [1.05], "AdjClose": [1.04], "Volume": [100]
        })}

    # Monkeypatch in module namespace
    import src.fetch_history_pro as local_fh
    import src.make_rebased_windows as local_mr

    from importlib import reload
    reload(local_fh)

    local_fh.batch_download = fake_batch

    fetch_out = tmp_path / "data"
    _run_script(local_fh, [
    "src.fetch_history_pro", "--input", str(tickers_csv), "--out", str(fetch_out), "--batch-size", "5"
    ])

    meta_fetch = fetch_out / "_meta.json"
    assert meta_fetch.exists()
    data_fetch = json.loads(meta_fetch.read_text(encoding="utf-8"))
    assert data_fetch.get("component") == "fetch"

    # Features
    feats_out = tmp_path / "features.parquet"
    _run_script(cf, [
    "src.compute_features", "--input", str(fetch_out / "history_all.parquet"), "--out", str(feats_out), "--csv", str(tmp_path / "features.csv")
    ])
    meta_features = (feats_out if feats_out.is_dir() else feats_out.parent) / "_meta.json"
    assert meta_features.exists()
    data_feats = json.loads(meta_features.read_text(encoding="utf-8"))
    assert data_feats.get("component") == "features"

    # Rebased
    rebased_out = tmp_path / "rebased"
    _run_script(mr, [
    "src.make_rebased_windows", "--input", str(feats_out), "--out", str(rebased_out), "--before", "3", "--after", "1"
    ])
    meta_rebased = rebased_out / "_meta.json"
    assert meta_rebased.exists()
    data_rebased = json.loads(meta_rebased.read_text(encoding="utf-8"))
    assert data_rebased.get("component") == "rebased"
