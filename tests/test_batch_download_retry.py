import pandas as pd
import pytest
import types

import fetch_history_pro as fh


class DummyError(Exception):
    pass


def test_batch_download_retries(monkeypatch):
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        # Simulate two failures then a valid response for tickers list
        calls["n"] += 1
        if calls["n"] < 3:
            raise DummyError("simulated network error")
        # return a simple dataframe structured like yfinance would for 1 ticker
        tickers = kwargs.get("tickers")
        if isinstance(tickers, list) and len(tickers) == 1:
            t = tickers[0]
            # produce a MultiIndex-like mapping when group_by='ticker' is used in yf.download
            df = pd.DataFrame({
                (t, 'Open'): [1.0, 2.0],
                (t, 'High'): [1.1, 2.1],
                (t, 'Low'): [0.9, 1.9],
                (t, 'Close'): [1.05, 2.05],
                (t, 'Adj Close'): [1.02, 2.02],
                (t, 'Volume'): [100, 200],
            }, index=pd.date_range('2024-01-01', periods=2))
            df.index.name = 'Date'
            # yfinance returns DataFrame with top-level columns when group_by='ticker'
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            return df
        # fallback
        return pd.DataFrame()

    monkeypatch.setattr(fh.yf, 'download', fake_download)

    res = fh.batch_download(['FOO'])
    assert 'FOO' in res
    df = res['FOO']
    # normalize_prices returns DataFrame with Ticker column
    assert 'Ticker' in df.columns
    assert df['Ticker'].iloc[0] == 'FOO'
    assert calls['n'] >= 3
