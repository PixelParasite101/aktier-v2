import pandas as pd
import numpy as np
from pathlib import Path
import subprocess, sys, json

# Denne test sikrer at benchmark scriptet kan køre i et minimalt miljø
# Forudsætter at der allerede findes rebased filer og index i repo'et.
# Hvis ikke, skip test.

def test_benchmark_minimal(tmp_path: Path):
    # Kræv at index og rebased mappe er til stede
    idx_path = Path('rebased_index.parquet')
    ref_dir = Path('rebased')
    if not idx_path.exists() or not ref_dir.exists():
        import pytest
        pytest.skip('rebased_index.parquet eller rebased/ findes ikke – skip benchmark test')

    # Tag første ticker i watch.csv
    watch = Path('watch.csv')
    if not watch.exists():
        import pytest
        pytest.skip('watch.csv findes ikke – skip benchmark test')

    # Kør kun index-vectorized med meget lille topk for hastighed
    cmd = [sys.executable, 'scripts/benchmark_prefilter.py', '--mode', 'index-vectorized', '--topk', '5', '--lookback', '75', '--metric', 'mse']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    # Auto snapshot bør være skrevet hvis --json-out ikke angives
    snap_dir = Path('analog_out/benchmark_runs')
    assert snap_dir.exists(), 'Snapshot dir blev ikke oprettet'
    snaps = list(snap_dir.glob('bench_*.json'))
    assert snaps, 'Ingen snapshot JSON fundet'
    # Læs seneste
    latest = max(snaps, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text())
    assert 'results' in data and data['results'], 'Result payload tom'
    # Sikre at valgt mode er korrekt
    assert data.get('mode') == 'index-vectorized'
    # Valider at topk felt reflekteres
    assert data.get('topk') == 5
