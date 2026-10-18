"""Benchmark the per-frame cost of each pipeline stage.

    python -m tools.bench            # CPU stages + NumPy renderer
    python -m tools.bench --gpu      # also the GPU renderer at projector sizes
"""

from __future__ import annotations

import argparse
import os
import time

import core  # noqa: F401  (sets OpenCV env before cv2 loads)
import cv2
import numpy as np

from core.engine import OrnamentEngine, load_tile
from core.posterize import PaletteTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timed(fn, repeat: int) -> float:
    fn()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat * 1000


def square(path: str) -> np.ndarray:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also benchmark the GPU renderer")
    args = ap.parse_args()

    img = square(os.path.join(ROOT, "me.jpg"))
    tile = load_tile(os.path.join(ROOT, "ornam.jpg"))

    for proc in (128, 256):
        tracker = PaletteTracker(k=4)
        small = cv2.resize(img, (proc, proc), interpolation=cv2.INTER_AREA)
        print(f"posterize proc={proc}: {timed(lambda: tracker.update(small, 1 / 30), 10):.1f} ms")

    tracker = PaletteTracker(k=4)
    labels, palette = tracker.update(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA), 1 / 30)
    for out in (512, 1024):
        t = time.perf_counter()
        eng = OrnamentEngine(tile, out_size=out, proc_size=128, max_level=12)
        build = time.perf_counter() - t
        eng.max_depth = 10
        upd = timed(lambda: eng.update(labels, palette, 1 / 30), 10)
        rend = timed(eng.render, 5)
        mp = out * out / 1e6
        print(f"numpy out={out}: build {build:.1f} s, update {upd:.1f} ms, render {rend:.1f} ms "
              f"({rend / mp:.0f} ms/MP), leaves {eng.leaf_count}")

    if args.gpu:
        from tools.bench_gpu import bench_gpu
        bench_gpu(tile, labels, palette)


if __name__ == "__main__":
    main()
