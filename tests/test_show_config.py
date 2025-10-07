import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"


def run_and_parse(args):
    # Run script with --show-config and parse JSON output
    cmd = [str(PY), str(ROOT / args), "--show-config"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Non-zero exit: {proc.stderr}"
    out = proc.stdout.strip()
    data = json.loads(out)
    return data


def test_fetch_history_show_config():
    cfg = run_and_parse("fetch_history_pro.py")
    # Expect some keys
    assert "preset" in cfg
    assert "input" in cfg or cfg.get("preset") is not None


def test_compute_features_show_config():
    cfg = run_and_parse("compute_features.py")
    assert "preset" in cfg
    assert "input" in cfg or cfg.get("preset") is not None


def test_make_rebased_show_config():
    cfg = run_and_parse("make_rebased_windows.py")
    assert "preset" in cfg
    assert "input" in cfg or cfg.get("preset") is not None
