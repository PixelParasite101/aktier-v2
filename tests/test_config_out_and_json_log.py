import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"


def run_cmd(args):
    cmd = [str(PY), str(ROOT / args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def test_show_config_writes_file(tmp_path: Path):
    out = tmp_path / "cfg.json"
    proc = subprocess.run([str(PY), str(ROOT / "fetch_history_pro.py"), "--show-config", "--config-out", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "preset" in data


def test_log_file_is_json_lines(tmp_path: Path):
    lf = tmp_path / "log.jsonl"
    # run show-config but set log-file so init_logging writes a JSON-lines header on actions
    proc = subprocess.run([str(PY), str(ROOT / "fetch_history_pro.py"), "--show-config", "--log-file", str(lf)], capture_output=True, text=True)
    assert proc.returncode == 0
    # file should exist (even if empty)
    assert lf.exists()
    # read lines; each should be valid JSON if non-empty
    content = lf.read_text(encoding="utf-8").strip()
    if content:
        for line in content.splitlines():
            json.loads(line)
