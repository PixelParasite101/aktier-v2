
import logging
import sys
import json
from typing import Optional


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # include extra if present
        if hasattr(record, "extra"):
            obj.update(record.extra)
        return json.dumps(obj, ensure_ascii=False)


def get_logger(name: str = __name__, level: int = logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def init_logging(log_file: Optional[str], level: int = logging.INFO):
    """Initialize root logger. If log_file is provided, add a JSON-lines file handler.

    The console handler is kept for human-readable output.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # ensure console handler exists
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(JsonLineFormatter())
        root.addHandler(fh)


# Convenience root logger
logger = get_logger("aktier", logging.INFO)
