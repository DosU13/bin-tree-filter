"""Deterministic OrnamentEngine scenarios shared by the golden-image test and its generator.

tests/data/engine_golden.npz was produced by tools/make_engine_golden.py with the
single-square engine as it was before the multi-cell rewrite; the rewrite must
reproduce it exactly.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "data", "engine_golden.npz")

# Fixed BGR palette; labels are nearest-palette-colour, so no k-means randomness.
PALETTE = np.array([[40, 30, 20], [60, 60, 170], [80, 160, 220], [220, 230, 240]], np.float32)

CASES = {
    "a": dict(out=96, proc=32, levels=8, crop=False, polarity="alternate", frames=12,
              settings=dict(max_depth=7, threshold=0.75), image="me.jpg", k=4),
    "b": dict(out=80, proc=32, levels=8, crop=True, polarity="invert", frames=6,
              settings=dict(max_depth=8, min_depth=2, threshold=0.9, conforming=False, animate=False),
              image="test_girl_sq.jpg", k=4),
    "c": dict(out=64, proc=64, levels=10, crop=False, polarity="none", frames=6,
              settings=dict(max_depth=9, threshold=0.95), image="aziz.jpg", k=2),
}


def frame_labels(image: str, proc: int, k: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels for one frame: the image drifts a few pixels per frame."""
    img = cv2.imread(os.path.join(ROOT, image))
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    img = np.roll(img, (frame * 3, frame * 2), axis=(0, 1))
    small = cv2.resize(img, (proc, proc), interpolation=cv2.INTER_AREA).astype(np.float32)
    pal = PALETTE if k == 4 else PALETTE[[0, 3]]
    d = ((small[:, :, None, :] - pal[None, None]) ** 2).sum(axis=3)
    return d.argmin(axis=2).astype(np.int64), pal


def run_case(name: str, engine_cls, load_tile) -> dict[str, np.ndarray]:
    """Run one scenario through an OrnamentEngine-compatible class; return its outputs."""
    c = CASES[name]
    tile = load_tile(os.path.join(ROOT, "ornam.jpg"), crop=c["crop"])
    eng = engine_cls(tile, out_size=c["out"], proc_size=c["proc"], max_level=c["levels"],
                     polarity=c["polarity"])
    for key, value in c["settings"].items():
        setattr(eng, key, value)
    frames = []
    for f in range(c["frames"]):
        labels, pal = frame_labels(c["image"], c["proc"], c["k"], f)
        eng.update(labels, pal, 1 / 30)
        frames.append(eng.render())
    return {
        "frames": np.stack(frames),
        "cord": eng.render(cord=True),
        "wire": eng.render(wireframe=True),
        "leaves": np.array(eng.leaf_count),
    }
