"""
Profile make_windows_for_ticker on the largest ticker group in the features dataset.

Usage: run from project root (script does not require args):
    venv\Scripts\python.exe scripts\profile_make_windows.py

Outputs a short pstats summary to stdout and writes full profile to ./profile_make_windows.prof
"""
from pathlib import Path
import cProfile
import pstats
import io
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_rebased_windows import load_data, make_windows_for_ticker


def main():
    # Try to load features from default folder
    data_path = Path("data") / "features"
    if not data_path.exists():
        # fallback to file
        data_path = Path("data")

    print(f"Loading data from: {data_path}")
    df = load_data(str(data_path))

    # pick largest ticker by rows
    groups = list(df.groupby("Ticker", sort=False, observed=False))
    if not groups:
        print("No ticker groups found in data")
        return

    ticker, g = max(groups, key=lambda kv: len(kv[1]))
    print(f"Profiling ticker: {ticker} with {len(g)} rows")

    # Profile a single call
    pr = cProfile.Profile()
    pr.enable()
    res = make_windows_for_ticker(g, before=20, after=5, drop_policy="any", require_full_window=True)
    pr.disable()

    out = io.StringIO()
    ps = pstats.Stats(pr, stream=out).sort_stats("cumulative")
    ps.print_stats(40)
    summary = out.getvalue()
    print(summary)

    # write full profile
    prof_path = Path("profile_make_windows.prof")
    pr.dump_stats(str(prof_path))
    print(f"Wrote full profile to: {prof_path}")


if __name__ == "__main__":
    main()
