"""
Quick checks that `_vectorized_make_windows` and
`_iterative_make_windows_optimized` produce identical outputs for several
synthetic datasets and parameter combinations.

Run with:
    venv\Scripts\python.exe scripts\check_windows_consistency.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_rebased_windows import (
    _vectorized_make_windows,
    _iterative_make_windows_optimized,
    order_columns,
)

from pandas.testing import assert_frame_equal


def make_synthetic(ticker="TST", n=30, seed=0, make_nans=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    adj = np.linspace(100, 150, n) + rng.normal(0, 1.0, size=n)
    close = adj + rng.normal(0, 0.5, size=n)
    ma10 = pd.Series(adj).rolling(10, min_periods=1).mean().to_numpy()
    ma20 = pd.Series(adj).rolling(20, min_periods=1).mean().to_numpy()
    rsi14 = 50 + rng.normal(0, 5, size=n)

    if make_nans:
        # introduce some NaNs
        adj[[3, 7, 12]] = np.nan
        close[[5, 12]] = np.nan
        ma10[[2, 15]] = np.nan

    df = pd.DataFrame(
        {
            "Ticker": [ticker] * n,
            "Date": dates,
            "AdjClose": adj,
            "Close": close,
            "MA_10": ma10,
            "MA_20": ma20,
            "RSI_14": rsi14,
        }
    )
    return df


def compare(df, before, after, drop_policy, require_full_window):
    a = _vectorized_make_windows(df, before, after, drop_policy, require_full_window)
    b = _iterative_make_windows_optimized(df, before, after, drop_policy, require_full_window)

    # order columns similarly
    a = order_columns(a)
    b = order_columns(b)

    try:
        assert_frame_equal(a, b, check_dtype=False, check_like=True, rtol=1e-6, atol=1e-8)
        return True, None
    except AssertionError as e:
        return False, str(e)


def run_checks():
    tests = []
    # simple small dataset
    tests.append((make_synthetic(n=30, seed=1, make_nans=False), 3, 2, "any", False))
    # bigger
    tests.append((make_synthetic(n=60, seed=2, make_nans=True), 5, 4, "any", False))
    # require full window
    tests.append((make_synthetic(n=40, seed=3, make_nans=True), 4, 4, "any", True))
    # different drop policies
    tests.append((make_synthetic(n=50, seed=4, make_nans=True), 6, 3, "all", False))
    tests.append((make_synthetic(n=35, seed=5, make_nans=True), 3, 3, "none", False))

    all_ok = True
    for i, (df, before, after, drop_policy, require_full_window) in enumerate(tests, 1):
        ok, msg = compare(df, before, after, drop_policy, require_full_window)
        params = f"before={before}, after={after}, drop_policy={drop_policy}, require_full_window={require_full_window}"
        if ok:
            print(f"Test {i} PASS ({params})")
        else:
            print(f"Test {i} FAIL ({params})")
            print(msg)
            all_ok = False

    return all_ok


if __name__ == "__main__":
    ok = run_checks()
    if not ok:
        print("Consistency checks failed")
        sys.exit(2)
    print("All consistency checks passed")
    sys.exit(0)
