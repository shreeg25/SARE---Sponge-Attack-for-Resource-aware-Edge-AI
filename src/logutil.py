"""Mirror everything a script prints (stdout and stderr, tracebacks included)
into a timestamped log file under results/logs/. Standard library only, so the
devkit environment can use it too."""

import datetime
import sys
from pathlib import Path


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(root, tag):
    """Start logging to <root>/results/logs/<tag>_<YYYYmmdd_HHMMSS>.log.
    Returns the log path."""
    log_dir = Path(root) / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    path = log_dir / f"{tag}_{now:%Y%m%d_%H%M%S}.log"
    fh = open(path, "a", encoding="utf-8", buffering=1)
    fh.write(f"# started {now.isoformat(timespec='seconds')}\n# cmd: {' '.join(sys.argv)}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"logging to {path}")
    return path
