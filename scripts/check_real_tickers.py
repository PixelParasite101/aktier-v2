"""
Run consistency and timing checks on a few real tickers from data/features.

Usage:
    venv\Scripts\python.exe scripts\check_real_tickers.py

Prints per-ticker pass/fail and timings.
"""
from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_rebased_windows import (
    _vectorized_make_windows,
    _iterative_make_windows_optimized,
    order_columns,
    load_data,
)

from pandas.testing import assert_frame_equal


def test_ticker(g, before, after, drop_policy, require_full_window):
    # time vectorized
    t0 = time.perf_counter()
    try:
        a = _vectorized_make_windows(g, before, after, drop_policy, require_full_window)
        vec_ok = True
    except Exception as e:
        a = None
        vec_ok = False
        vec_exc = e
    t1 = time.perf_counter()
    vec_time = t1 - t0

    t0 = time.perf_counter()
    b = _iterative_make_windows_optimized(g, before, after, drop_policy, require_full_window)
    t1 = time.perf_counter()
    it_time = t1 - t0

    status = ""
    if not vec_ok:
        status = f"VECTOR_FAIL: {vec_exc}"
    else:
        a_o = order_columns(a)
        b_o = order_columns(b)
        try:
            assert_frame_equal(a_o, b_o, check_dtype=False, check_like=True, rtol=1e-6, atol=1e-8)
            status = "PASS"
        except AssertionError as e:
            status = f"MISMATCH: {e}"

    return status, vec_time, it_time, len(g)


def main():
    data_path = Path("data") / "features"
    print(f"Loading data from {data_path}")
    df = load_data(str(data_path))
    counts = df["Ticker"].value_counts()
    top = counts.head(6).index.tolist()
    print(f"Top tickers: {top}")

    results = []
    for t in top:
        g = df[df["Ticker"] == t].reset_index(drop=True)
        print(f"Testing {t} ({len(g)} rows)")
        status, vt, it, nrows = test_ticker(g, before=20, after=5, drop_policy="any", require_full_window=True)
        results.append((t, nrows, status, vt, it))
        print(f"  -> {status}; vectorized={vt:.3f}s iterative={it:.3f}s")

    print("\nSummary:")
    for t, nrows, status, vt, it in results:
        print(f"{t}: rows={nrows} status={status} vec={vt:.3f}s it={it:.3f}s")


if __name__ == "__main__":
    main()
