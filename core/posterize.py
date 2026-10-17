"""Temporally stable posterization for video.

posterize.py runs a fresh K-means (10 random restarts) on every frame, so the
palette and the cluster order can jump from one frame to the next — even on an
identical frame.  Here the centres are found once and then *tracked*: each
frame does one Lloyd step seeded from the previous centres and blends it in
with an EMA, so cluster identities never swap.  A per-pixel margin keeps a
pixel's previous label unless another centre is clearly closer, which kills
sensor-noise flicker on colour boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np


# Dark -> light ramps (RGB hex).  Clusters are mapped onto a ramp by lightness.
PALETTE_STYLES: dict[str, list[str] | None] = {
    "natural": None,
    "navy-red-gold": ["#14213d", "#9b2226", "#e09f3e", "#f5e6c8"],
    "shyrdak": ["#2b1d14", "#7a2e1c", "#c46a2b", "#e8c39e"],
    "tush-kiyiz": ["#0b3c49", "#c1121f", "#f4a259", "#fdf0d5"],
}


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.reshape(1, -1, 3).astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr.reshape(-1, 3) * 255, 0, 255).astype(np.float32)


def _ramp_lab(hexes: list[str], k: int) -> np.ndarray:
    """Resample a hex ramp to *k* colours, interpolating in Lab."""
    rgb = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], np.float32)
    lab = cv2.cvtColor(rgb[None, :, ::-1] / 255, cv2.COLOR_BGR2Lab)[0]
    pos = np.linspace(0, len(hexes) - 1, k)
    return np.stack([np.interp(pos, np.arange(len(hexes)), lab[:, c]) for c in range(3)], axis=1)


class PaletteTracker:
    """K-means posterizer whose centres are tracked across frames."""

    def __init__(self, k: int = 4, tau: float = 0.6, label_margin: float = 4.0) -> None:
        self.k = k
        self.tau = tau                    # seconds for centres to follow the scene
        self.label_margin = label_margin  # Lab units a pixel must gain to switch label
        self.style = "natural"
        self.centers: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def reset(self) -> None:
        self.centers = None
        self._labels = None

    def update(self, bgr: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Posterize one frame.

        Returns:
            labels:  (H, W) int64 cluster index per pixel.
            palette: (k, 3) float32 BGR display colour per cluster.
        """
        lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        x = lab.reshape(-1, 3)

        if self.centers is None or len(self.centers) != self.k:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _, _, c = cv2.kmeans(x, self.k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            self.centers = c[np.argsort(c[:, 0])]
            self._labels = None

        c = self.centers
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        rows = np.arange(len(x))

        if self._labels is not None and self._labels.shape == labels.shape:
            prev = self._labels
            keep = np.sqrt(d[rows, prev]) <= np.sqrt(d[rows, labels]) + self.label_margin
            labels = np.where(keep, prev, labels)

        # One Lloyd step from the previous centres, blended in with an EMA
        counts = np.bincount(labels, minlength=self.k)
        sums = np.stack([np.bincount(labels, weights=x[:, i], minlength=self.k) for i in range(3)], axis=1)
        new_c = c.copy()
        has = counts > 0
        new_c[has] = sums[has] / counts[has, None]
        # Re-seed a dead cluster at the worst-represented pixel
        for j in np.flatnonzero(~has):
            new_c[j] = x[d.min(axis=1).argmax()]
        a = 1.0 if self.tau <= 0 else float(1.0 - np.exp(-dt / self.tau))
        self.centers = (c + a * (new_c - c)).astype(np.float32)
        self._labels = labels

        return labels.reshape(bgr.shape[:2]), self.palette()

    def palette(self) -> np.ndarray:
        """Display colours: tracked centres, or the active style ramp by lightness."""
        ramp = PALETTE_STYLES[self.style]
        if ramp is None:
            return _lab_to_bgr(self.centers)
        ramp_lab = _ramp_lab(ramp, self.k)
        rank = np.argsort(np.argsort(self.centers[:, 0]))
        return _lab_to_bgr(ramp_lab[rank])
