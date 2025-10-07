import subprocess
import sys
import tempfile
import time
from pathlib import Path


def test_dry_run_heartbeat_emits():
    """Run analog_matcher_watch.py in dry-run mode and assert heartbeat lines appear."""
    repo = Path(__file__).resolve().parents[1]
    script = repo / "analog_matcher_watch.py"
    # Prepare a minimal watch.csv
    watch = repo / "watch.csv"
    watch.write_text("Ticker\nAAPL\nMSFT\n")

    # Run the script for a short time and capture output
    cmd = [sys.executable, str(script), "--watch", str(watch), "--dry-run", "--heartbeat-interval", "0.5"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        seen_heartbeat = False
        start = time.time()
        out_lines = []
        # Read lines for up to 6 seconds
        while time.time() - start < 6:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            out_lines.append(line)
            if "heartbeat:" in line:
                seen_heartbeat = True
                break
        # Terminate the process
        proc.terminate()
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert seen_heartbeat, f"No heartbeat lines found in output:\n{''.join(out_lines)}"