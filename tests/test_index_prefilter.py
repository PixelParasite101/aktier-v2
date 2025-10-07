import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def _make_synthetic_rebased(tmpdir: Path, tickers=('AAA','BBB'), lookback=10, horizon=3):
    # Create minimal rebased parquet files with offsets -L+1..+H
    lb_start = -lookback + 1
    offsets = list(range(lb_start, horizon + 1))
    for t in tickers:
        rows = []
        for ref_i in range(2):  # two refdates per ticker
            refdate = pd.Timestamp('2024-01-0{}'.format(ref_i+1))
            base_series = np.linspace(90, 110, len(offsets)) + ref_i
            ma20 = base_series * 1.01
            ma50 = base_series * 0.99
            ma200 = base_series * 1.02
            for off, v, m20, m50, m200 in zip(offsets, base_series, ma20, ma50, ma200):
                rows.append({
                    'Ticker': t,
                    'RefDate': refdate,
                    'Offset': off,
                    'AdjClose_Rebased': v,
                    'MA_20_Rebased': m20,
                    'MA_50_Rebased': m50,
                    'MA_200_Rebased': m200,
                })
        df = pd.DataFrame(rows)
        df.to_parquet(tmpdir / f"{t}_rebased.parquet", index=False)


def test_prefilter_dry_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    rebased_dir = tmp_path / 'rebased'
    rebased_dir.mkdir()
    _make_synthetic_rebased(rebased_dir)

    # Build index
    index_file = tmp_path / 'rebased_index.parquet'
    cmd_index = [sys.executable, str(repo / 'reference_index.py'), '--rebased-dir', str(rebased_dir), '--out', str(index_file), '--lookback', '10', '--sig-downsample', '4', '--match-cols', 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased']
    subprocess.check_call(cmd_index)

    # Create watch.csv
    watch = tmp_path / 'watch.csv'
    watch.write_text('Ticker\nAAA\n')

    # Run analog matcher in dry-run mode with index (it will fall back to synthetic data fetch path but we only test prefilter logic up to scanning)
    out_dir = tmp_path / 'analog_out'
    cmd = [
        sys.executable, str(repo / 'analog_matcher_watch.py'),
        '--watch', str(watch),
        '--dry-run',
        '--use-index', '--index-file', str(index_file),
        '--ref-dir', str(rebased_dir),
        '--out-dir', str(out_dir),
        '--lookback', '10', '--horizon', '3', '--prefilter-topk', '5',
        '--match-cols', 'AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased'
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stderr
    # Check log contains Prefilter
    assert 'Prefilter:' in proc.stdout, proc.stdout
    assert 'Scan summary:' in proc.stdout, proc.stdout
