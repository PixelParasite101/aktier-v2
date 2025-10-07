import json
import subprocess
import sys
from pathlib import Path


def run_cmd(args):
    # Brug samme python exe
    exe = sys.executable
    proc = subprocess.run([exe, 'reference_index.py'] + args, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def test_show_config_standard_preset():
    code, out, err = run_cmd(['--preset', 'standard', '--show-config'])
    assert code == 0, f"Non-zero exit: {code} stderr={err}"
    cfg = json.loads(out)
    # Basis felter
    assert cfg['preset'] == 'standard'
    assert cfg['rebased_dir'] == 'rebased'
    assert cfg['out'] == 'rebased_index.parquet'
    assert cfg['lookback'] == 75
    assert cfg['sig_downsample'] == 8
    # match-cols default
    assert 'AdjClose_Rebased' in cfg['match_cols']


def test_show_config_auto_preset_when_no_args():
    # Kør uden args (kun scriptnavn) => auto preset = standard
    code, out, err = run_cmd(['--show-config'])
    assert code == 0, f"Non-zero exit: {code} stderr={err}"
    cfg = json.loads(out)
    assert cfg['preset'] == 'standard'
    assert cfg['rebased_dir'] == 'rebased'

