import json
import io
import sys
from contextlib import redirect_stdout

import fetch_history_pro as fh
import compute_features as cf


def run_show_config(module, argv):
    buf = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(buf):
            module.main()
    finally:
        sys.argv = old_argv
    data = json.loads(buf.getvalue())
    return data


def test_fetch_preset_override_batch_size():
    cfg = run_show_config(fh, ["fetch_history_pro.py", "--preset", "standard", "--batch-size", "5", "--show-config"])
    assert cfg["batch_size"] == 5, "User override of batch-size should persist over preset"


def test_compute_features_preset_override_ma():
    # override default MA set
    cfg = run_show_config(cf, ["compute_features.py", "--preset", "standard", "--ma", "10", "30", "--show-config"])
    assert cfg["ma"] == [10, 30]
