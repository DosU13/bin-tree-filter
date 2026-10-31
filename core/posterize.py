"""Temporally stable posterization for video.

posterize.py runs a fresh K-means (10 random restarts) on every frame, so the
palette and the cluster order can jump from one frame to the next — even on an
identical frame.  Here the centres are found once and then *tracked*: each
frame does one Lloyd step seeded from the previous centres and blends it in
with an EMA, so cluster identities never swap.  A per-pixel margin keeps a
pixel's previous label unless another centre is clearly closer, which kills
sensor-noise flicker on colour boundaries.

Depth mode's counterpart is DepthLayers: fixed slices of the depth band
instead of tracked clusters, coloured by depth_palette.
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
# Each style's main colour (its red): with motion glow on it lights the glow alone
PALETTE_MAIN: dict[str, str] = {
    "navy-red-gold": "#9b2226",
    "shyrdak": "#7a2e1c",
    "tush-kiyiz": "#c1121f",
}


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.reshape(1, -1, 3).astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr.reshape(-1, 3) * 255, 0, 255).astype(np.float32)


def _hex_lab(hexes: list[str]) -> np.ndarray:
    rgb = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], np.float32)
    return cv2.cvtColor(rgb[None, :, ::-1] / 255, cv2.COLOR_BGR2Lab)[0]


def _ramp_lab(hexes: list[str], k: int) -> np.ndarray:
    """Resample a hex ramp to *k* colours, interpolating in Lab."""
    lab = _hex_lab(hexes)
    pos = np.linspace(0, len(hexes) - 1, k)
    return np.stack([np.interp(pos, np.arange(len(hexes)), lab[:, c]) for c in range(3)], axis=1)


def split_main(style: str) -> tuple[list[str], str] | None:
    """A palette style's ramp without its main colour, and that colour (None without a ramp).

    With motion glow on, the main colour (PALETTE_MAIN, else the most saturated)
    is kept for the light alone: the posterize / depth colours use the rest.
    """
    ramp = PALETTE_STYLES.get(style)
    if not ramp:
        return None
    main = PALETTE_MAIN.get(style)
    if main not in ramp:
        lab = _hex_lab(ramp)
        main = ramp[int(np.argmax(np.hypot(lab[:, 1], lab[:, 2])))]
    return [c for c in ramp if c != main], main


def _ramp(style: str, glow: bool) -> list[str] | None:
    """The colours a style gives posterize / depth layers (without the glow's when glow is on)."""
    split = split_main(style) if glow else None
    return split[0] if split else PALETTE_STYLES.get(style)


class PaletteTracker:
    """K-means posterizer whose centres are tracked across frames."""

    def __init__(self, k: int = 4, tau: float = 0.6, label_margin: float = 4.0) -> None:
        self.k = k
        self.tau = tau                    # seconds for centres to follow the scene
        self.label_margin = label_margin  # Lab units a pixel must gain to switch label
        self.style = "natural"
        self.glow = False                 # keep the style's main colour for the motion glow
        self.centers: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def reset(self) -> None:
        self.centers = None
        self._labels = None

    def update(self, bgr: np.ndarray, dt: float, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Posterize one frame.

        Args:
            mask: Optional (H, W) bool; only these pixels shape the palette
                  (e.g. the parts of the stage some camera actually sees).
                  Every pixel still gets a label.

        Returns:
            labels:  (H, W) int64 cluster index per pixel.
            palette: (k, 3) float32 BGR display colour per cluster.
        """
        lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        x = lab.reshape(-1, 3)
        fit = None if mask is None else mask.reshape(-1)
        xf = x if fit is None else x[fit]
        if len(xf) < self.k:
            # Nothing (or almost nothing) to learn from: keep the current palette
            if self.centers is None or len(self.centers) != self.k:
                return np.zeros(bgr.shape[:2], dtype=np.int64), np.zeros((self.k, 3), np.float32)
            d = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
            return d.argmin(axis=1).reshape(bgr.shape[:2]), self.palette()

        if self.centers is None or len(self.centers) != self.k:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _, _, c = cv2.kmeans(np.ascontiguousarray(xf), self.k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
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
        lf = labels if fit is None else labels[fit]
        df = d if fit is None else d[fit]
        counts = np.bincount(lf, minlength=self.k)
        sums = np.stack([np.bincount(lf, weights=xf[:, i], minlength=self.k) for i in range(3)], axis=1)
        new_c = c.copy()
        has = counts > 0
        new_c[has] = sums[has] / counts[has, None]
        # Re-seed a dead cluster at the worst-represented pixel
        for j in np.flatnonzero(~has):
            new_c[j] = xf[df.min(axis=1).argmax()]
        a = 1.0 if self.tau <= 0 else float(1.0 - np.exp(-dt / self.tau))
        self.centers = (c + a * (new_c - c)).astype(np.float32)
        self._labels = labels

        return labels.reshape(bgr.shape[:2]), self.palette()

    def palette(self) -> np.ndarray:
        """Display colours: tracked centres, or the active style ramp by lightness."""
        ramp = _ramp(self.style, self.glow)
        if ramp is None:
            return _lab_to_bgr(self.centers)
        ramp_lab = _ramp_lab(ramp, self.k)
        rank = np.argsort(np.argsort(self.centers[:, 0]))
        return _lab_to_bgr(ramp_lab[rank])


# ---------------------------------------------------------------------------
# Depth layers
# ---------------------------------------------------------------------------

DEPTH_PALETTES = ("fade",) + tuple(name for name, ramp in PALETTE_STYLES.items() if ramp)
DEPTH_MARGIN = 3.0      # grey levels a pixel must go past a layer's edge before it changes layer


def depth_palette(n: int, style: str, fg, bg, glow: bool = False) -> np.ndarray:
    """Colours for depth layers: (n + 1, 3) float32 BGR, the background first, then nearest to farthest.

    "fade" runs from the foreground colour towards the background, so far layers
    fade out.  A palette style runs from the end of its ramp least like the
    background in lightness towards the other end, so the nearest layer stands
    out most; with *glow* its main colour is left for the light.
    """
    bg = np.asarray(bg, np.float32)
    ramp = _ramp(style, glow)
    if ramp:
        lab = _ramp_lab(ramp, n)
        bg_light = cv2.cvtColor(bg.reshape(1, 1, 3) / 255, cv2.COLOR_BGR2Lab)[0, 0, 0]
        layers = _lab_to_bgr(lab[np.argsort(-np.abs(lab[:, 0] - bg_light), kind="stable")])
    else:
        t = (np.arange(n, dtype=np.float32) / n)[:, None]
        layers = np.asarray(fg, np.float32) * (1 - t) + bg * t
    return np.vstack([bg[None], layers]).astype(np.float32)


class DepthLayers:
    """Depth picture -> layer labels: the band [lo, hi] cut into n equal slices.

    Label 0 is outside the band, no reading or no camera; 1 is the slice nearest
    the camera (brightest) ... n the farthest.  A pixel keeps its label until it
    is DEPTH_MARGIN grey levels past the edge, so noise on a boundary doesn't
    flicker.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._labels: np.ndarray | None = None
        self._key = None

    def update(self, depth: np.ndarray, valid: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
        """Labels (H, W) int64 for *depth* (float 0..255, near bright) where *valid*."""
        n = max(1, int(n))
        width = max(hi - lo, 1e-6) / n
        pos = (np.float32(hi) - depth.astype(np.float32, copy=False)) * np.float32(1 / width)   # 0 at hi, n at lo
        labels = np.clip(pos, 0, n - 1).astype(np.uint8) + np.uint8(1)
        labels[~(valid & (pos >= 0) & (pos <= n))] = 0
        key = (depth.shape, lo, hi, n)
        prev = self._labels
        if prev is not None and key == self._key:
            # Keep the previous label while within the margin of its slice (in slice units)
            m = np.float32(min(DEPTH_MARGIN / width, 0.25))
            keep = np.abs(pos - (prev.astype(np.float32) - np.float32(0.5))) <= np.float32(0.5) + m
            keep &= valid & (prev > 0)
            keep |= (prev == 0) & ~(valid & (pos >= m) & (pos <= n - m))
            np.copyto(labels, prev, where=keep)
        self._labels, self._key = labels, key
        return labels.astype(np.int64)
