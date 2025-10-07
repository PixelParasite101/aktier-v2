import pandas as pd
import pytest
import sys
import os
from pathlib import Path
import fetch_history_pro as fh


class DummyArgs:
    pass


def test_fail_on_empty(monkeypatch, tmp_path):
    # Prepare an input CSV with one ticker
    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("ticker\nNO_DATA\n", encoding="utf-8")

    # Monkeypatch batch_download to return empty dataframe for ticker
    def fake_batch_download(tickers):
        return {t: pd.DataFrame() for t in tickers}

    monkeypatch.setattr(fh, "batch_download", fake_batch_download)

    # Run main with fail-on-empty=0 expecting exit code 1
    argv = [
        "fetch_history_pro.py",
        "--input", str(csv_path),
        "--out", str(tmp_path / "out"),
        "--fail-on-empty", "0",
        "--batch-size", "1",
        "--show-config"  # We'll first verify config prints fine
    ]
    # show-config path: should not raise
    old = sys.argv
    sys.argv = argv
    try:
        fh.main()
    finally:
        sys.argv = old

    # Now actual run (remove --show-config)
    argv_run = [
        "fetch_history_pro.py",
        "--input", str(csv_path),
        "--out", str(tmp_path / "out2"),
        "--fail-on-empty", "0",
        "--batch-size", "1"
    ]
    sys.argv = argv_run
    with pytest.raises(SystemExit) as exc:
        fh.main()
    sys.argv = old
    assert exc.value.code == 1
