import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

from tests.test_index_prefilter import _make_synthetic_rebased  # reuse helper

def test_prefilter_corr(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    rebased_dir = tmp_path / 'rebased'
    rebased_dir.mkdir()
    _make_synthetic_rebased(rebased_dir)

    index_file = tmp_path / 'rebased_index.parquet'
    cmd_index = [sys.executable, str(repo / 'reference_index.py'), '--rebased-dir', str(rebased_dir), '--out', str(index_file), '--lookback', '10', '--sig-downsample', '4']
    subprocess.check_call(cmd_index)

    watch = tmp_path / 'watch.csv'
    watch.write_text('Ticker\nAAA\n')
    out_dir = tmp_path / 'analog_out'
    cmd = [
        sys.executable, str(repo / 'analog_matcher_watch.py'),
        '--watch', str(watch), '--dry-run', '--use-index', '--index-file', str(index_file),
        '--ref-dir', str(rebased_dir), '--out-dir', str(out_dir), '--lookback', '10', '--horizon', '3',
        '--prefilter-topk', '5', '--metric', 'corr'
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert 'Prefilter:' in proc.stdout
    assert 'Scan summary:' in proc.stdout
