"""Per-crop cost of the appearance embedder on this device's CPU (run it on the
laptop now and on the Pi 5 later). The median ms/crop feeds replay_pipeline.py
--reid-ms, which turns ReID call counts into time.

    python scripts/bench_reid.py
"""

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.logutil import start_log
from src.reid import Embedder


def main():
    start_log(ROOT, "bench_reid")
    torch.manual_seed(0)
    model = Embedder().eval()
    print(f"CPU threads: {torch.get_num_threads()}")
    for n in (1, 4, 16, 32):
        x = torch.rand(n, 3, 128, 128)
        for _ in range(3):
            model(x)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(times)
        print(f"batch {n:3d}: {med:8.2f} ms total, {med / n:6.2f} ms/crop")


if __name__ == "__main__":
    main()
