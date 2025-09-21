Title: Add CI, developer docs and optimize build_rebased_index parquet reads

Description:

This PR bundles three quick improvements to improve developer experience and index build performance:

1) CI: Add GitHub Actions workflow (.github/workflows/ci.yml)
   - Runs on push and pull_request to `main`.
   - Matrix: Python 3.11 and 3.10
   - Steps: checkout, setup-python, install requirements, run `ruff` and `pytest`.

2) README: Developer section
   - Adds a Developer section with quick-start instructions (PowerShell), a test command (`pytest`), example standard pipeline, and a CI status badge.
   - Adds a short compatibility note about `pandas`/`pyarrow` versions and Parquet I/O.

3) scripts/build_rebased_index.py: I/O optimization
   - Avoid reading the same Parquet file twice when trying to project columns.
   - Uses `pyarrow.parquet.ParquetFile` (when available) to inspect columns, then reads only the needed columns in a single `pd.read_parquet(...)` call. Falls back to a single full read when pyarrow is not available or projection fails.

Testing performed locally:
- Ran pytest: `python -m pytest -q` → 8 passed
- Ran `scripts/build_rebased_index.py` locally to build `rebased_index.parquet` successfully

Notes and follow-ups:
- Consider adding a CI status badge in README (already added, points to the new workflow). Adjust repo path if needed.
- Consider storing downsampled signatures as native Parquet list/array types for faster queries (future work).
- Consider adding an end-to-end smoke test for features→rebased→index flows.

Suggested reviewers: @PixelParasite101 (owner)

How to test locally:
- Create venv and install requirements:
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
- Run tests:
  python -m pytest -q
- Run the index builder:
  & .\venv\Scripts\python.exe scripts\build_rebased_index.py

If you'd like, I can create a branch and open the PR on your behalf (requires push permissions).
