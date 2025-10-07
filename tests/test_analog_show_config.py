import json
import subprocess
import sys
from pathlib import Path

def run_cmd(args):
    exe = sys.executable
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run([exe, 'analog_matcher_watch.py'] + args, capture_output=True, text=True, cwd=repo_root)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def test_analog_show_config_preset_standard_no_index():
    # Sikrer at auto index ikke fejler når filen ikke findes
    code, out, err = run_cmd(['--preset', 'standard', '--show-config', '--watch', 'watch.csv'])
    assert code == 0, f"Exit code {code} stderr={err}"
    cfg = json.loads(out)
    assert cfg['preset'] == 'standard'
    # Hvis index ikke findes skal use_index muligvis være False
    assert 'use_index' in cfg


def test_analog_show_config_auto_preset():
    # Ingen args -> auto preset standard
    code, out, err = run_cmd(['--show-config', '--watch', 'watch.csv'])
    assert code == 0, f"Exit code {code} stderr={err}"
    cfg = json.loads(out)
    assert cfg['preset'] == 'standard'
    assert cfg['watch'] == 'watch.csv'
