"""Motion glow: movement lights the ornament up in a bright colour, and a small
fluid simulation carries the light on, so a waving hand sends it further.

    MotionGlow.update(image, valid, dt, ...) -> glow, float32 0..1 at stage proc resolution
    glow_labels(labels, palette, glow, colour, background) -> labels, palette with the glow painted in

Where the light comes from (`source`):
    motion   change from the last few frames: whatever moves
    person   difference from the environment, which is each pixel's most common
             value over the last `memory` seconds (a decaying histogram).  It
             learns itself while people play, and whoever is in front of it
             stays lit while they stand still (until they stay longer than that)
    both     the person at half strength, their movement at full

The light is dye in a coarse "stable fluids" simulation (Stam 1999): movement
measured by optical flow pushes the fluid, the fluid carries the dye on and
swirls, and the dye fades over `fade` seconds.

Depth pictures (depth mode: near bright, 0 = no reading) count only what is
nearer than the environment and skip pixels without a reading; colour
pictures (their brightness) count any difference.

Light thinner than SPECK of a cell is dropped (flickering edges, sensor noise).
Cost: a few ms a frame; the fluid and the optical flow run at 32 px per cell
and the environment at 64, whatever the statistics resolution.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

BINS = 32                  # environment histogram: 256 / BINS grey levels per bin
ENV_PX = 64                # environment: pixels per stage cell
FLOW_PX = 32               # optical flow: pixels per stage cell
FLUID_PX = 32              # fluid grid: cells per stage cell
SPECK = 1 / 24             # light thinner than this share of a cell is noise (flickering edges)
ENV_EVERY = 4              # recompute the environment's most common values every this many frames
RECENT_TAU = 0.15          # s: movement = difference from the picture of about this long ago
FLOW_TAU = 1.0             # s: how long the fluid keeps moving once nothing pushes it
PROJECT_STEPS = 20         # pressure solver iterations (keeps the flow swirling, not piling up)
LEVELS = (0.15, 0.4, 0.7)  # glow above these: the glow colour at 1/3, 2/3 and full strength

_DX = np.array([[-0.5, 0.0, 0.5]], np.float32)
_DY = _DX.T.copy()
_CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)


def _project(u: np.ndarray, v: np.ndarray) -> None:
    """Make the velocity field divergence-free in place (Jacobi pressure solve)."""
    div = cv2.filter2D(u, -1, _DX, borderType=cv2.BORDER_CONSTANT) + \
        cv2.filter2D(v, -1, _DY, borderType=cv2.BORDER_CONSTANT)
    p = np.zeros_like(div)
    for _ in range(PROJECT_STEPS):
        p = (cv2.filter2D(p, -1, _CROSS, borderType=cv2.BORDER_REPLICATE) - div) * 0.25
    u -= cv2.filter2D(p, -1, _DX, borderType=cv2.BORDER_REPLICATE)
    v -= cv2.filter2D(p, -1, _DY, borderType=cv2.BORDER_REPLICATE)


class MotionGlow:
    """Light from movement or people, carried by a fluid (see the module docstring)."""

    def __init__(self) -> None:
        self.glow: np.ndarray | None = None
        self.forget_environment()
        self.clear()

    def clear(self) -> None:
        """Drop the light, the fluid and the motion memory (keeps the environment)."""
        self._key = None
        self._recent: np.ndarray | None = None
        self._prev: np.ndarray | None = None      # last distinct picture at flow resolution (uint8)
        self._prev_time = 0.0
        self._clock = 0.0
        self._u = self._v = self._dye = None

    def forget_environment(self) -> None:
        """Learn the environment afresh (the cameras moved)."""
        self._hist: np.ndarray | None = None
        self._weight = 1.0
        self._env: np.ndarray | None = None
        self._frames = 0

    # -- per frame -----------------------------------------------------------

    def update(self, image: np.ndarray, valid: np.ndarray, dt: float, source: str = "motion",
               depth: bool = True, threshold: float = 8.0, memory: float = 30.0, flow: float = 1.0,
               fade: float = 1.5, cell_px: int = 128) -> np.ndarray:
        """One frame.

        Args:
            image:     stage picture at proc resolution: grey depth (float or uint8) or BGR
            valid:     bool, pixels with a reading (depth) / seen by a camera
            source:    "motion", "person" or "both"
            depth:     image is a depth picture (near bright, 0 = no reading)
            threshold: grey-level change that counts
            memory:    seconds the environment remembers (source person / both)
            flow:      how strongly movement pushes the fluid (0 = the light stays put)
            fade:      time constant (s) of the light
            cell_px:   proc pixels per stage cell
        """
        g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        h, w = g.shape
        key = (h, w, cell_px)
        if key != self._key:
            self.clear()
            self._key = key
            fw, fh = max(2, round(w * FLUID_PX / cell_px)), max(2, round(h * FLUID_PX / cell_px))
            self._u = np.zeros((fh, fw), np.float32)
            self._v = np.zeros((fh, fw), np.float32)
            self._dye = np.zeros((fh, fw), np.float32)
            xs, ys = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
            self._grid = (xs, ys)
            k = max(3, round(cell_px * SPECK) | 1)
            self._speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._clock += dt

        light = np.zeros((h, w), np.float32)
        if source in ("motion", "both"):
            light = self._motion(g, valid, dt, threshold, depth)
        if source in ("person", "both"):
            person = self._person(g, valid, dt, threshold, memory, depth, cell_px)
            np.maximum(light, person * (0.5 if source == "both" else 1.0), out=light)
        light = cv2.morphologyEx(light, cv2.MORPH_OPEN, self._speck)
        self._fluid(g, light, dt, flow, fade, cell_px)
        dye = cv2.resize(self._dye, (w, h), interpolation=cv2.INTER_LINEAR)
        self.glow = np.maximum(light, np.clip(dye, 0.0, 1.0))
        return self.glow

    def _motion(self, g, valid, dt, threshold, depth) -> np.ndarray:
        if self._recent is None:
            self._recent = g.copy()
        recent = self._recent
        seen = valid & (recent > 0) if depth else valid
        light = np.clip((cv2.absdiff(g, recent) - threshold) / max(threshold, 1.0), 0.0, 1.0)
        light[~seen] = 0.0
        cv2.accumulateWeighted(g, recent, 1.0 - math.exp(-dt / RECENT_TAU), mask=valid.view(np.uint8))
        if depth:
            np.copyto(recent, g, where=valid & ~seen)          # first reading: nothing to compare yet
        return light

    def _person(self, g, valid, dt, threshold, memory, depth, cell_px) -> np.ndarray:
        h, w = g.shape
        sw, sh = max(1, round(w * ENV_PX / cell_px)), max(1, round(h * ENV_PX / cell_px))
        small = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_NEAREST)     # no averaging across holes
        bins = (np.clip(small, 0, 255) * (BINS / 256)).astype(np.int32).ravel()
        if self._hist is None or self._hist.shape[0] != bins.size:
            self.forget_environment()
            self._hist = np.zeros((bins.size, BINS), np.float32)
        # Decaying histogram: newer frames weigh exp(age / memory) more (renormalised now and then)
        self._weight *= math.exp(dt / max(memory, 1e-3))
        if self._weight > 1e6:
            self._hist /= self._weight
            self._weight = 1.0
        self._hist[np.arange(bins.size), bins] += self._weight
        if self._env is None or self._env.shape != (h, w) or self._frames % ENV_EVERY == 0:
            top = (self._hist.argmax(axis=1).reshape(sh, sw) + 1) * np.float32(256 / BINS)   # top of the bin
            self._env = cv2.resize(top, (w, h), interpolation=cv2.INTER_NEAREST)
        self._frames += 1
        half = 128 / BINS
        diff = g - self._env if depth else np.abs(g - (self._env - half)) - half
        return (valid & (diff > threshold)).astype(np.float32)

    def _fluid(self, g, light, dt, gain, fade, cell_px) -> None:
        u, v = self._u, self._v
        fh, fw = u.shape
        h, w = g.shape
        ow, oh = max(8, round(w * FLOW_PX / cell_px)), max(8, round(h * FLOW_PX / cell_px))
        pic = cv2.resize(np.clip(g, 0, 255), (ow, oh), interpolation=cv2.INTER_AREA).astype(np.uint8)
        src = cv2.resize(light, (fw, fh), interpolation=cv2.INTER_AREA)
        if self._prev is not None and gain > 0 and not np.array_equal(pic, self._prev):
            # Movement since the last new picture pushes the fluid where there is light
            span = max(self._clock - self._prev_time, 1e-3)
            f = cv2.calcOpticalFlowFarneback(self._prev, pic, None, 0.5, 2, 7, 2, 5, 1.1, 0)
            f = cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA) * np.float32(FLUID_PX / FLOW_PX / span * gain)
            k = np.clip(src * 2.0, 0.0, 1.0)
            u += (f[:, :, 0] - u) * k
            v += (f[:, :, 1] - v) * k
        if self._prev is None or not np.array_equal(pic, self._prev):
            self._prev, self._prev_time = pic, self._clock
        np.maximum(self._dye, src, out=self._dye)

        # Advect (semi-Lagrangian: fetch each cell's contents from where the flow brings them from)
        xs, ys = self._grid
        mx, my = xs - u * np.float32(dt), ys - v * np.float32(dt)
        warp = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        u, v = cv2.remap(u, mx, my, **warp), cv2.remap(v, mx, my, **warp)
        dye = cv2.remap(self._dye, mx, my, **warp)
        _project(u, v)
        keep = np.float32(math.exp(-dt / FLOW_TAU))
        self._u, self._v = u * keep, v * keep
        self._dye = cv2.GaussianBlur(dye, (3, 3), 0.6) * np.float32(math.exp(-dt / max(fade, 1e-3)))


def glow_labels(labels: np.ndarray, palette: np.ndarray, glow: np.ndarray, colour, background
                ) -> tuple[np.ndarray, np.ndarray]:
    """Paint the glow over posterize / depth labels.

    Pixels above LEVELS[i] get one of three new labels: the glow colour at 1/3,
    2/3 and full strength over the background colour.
    """
    level = (glow > LEVELS[0]).astype(np.int64)
    for t in LEVELS[1:]:
        level += glow > t
    k = len(palette)
    bg, c = np.asarray(background, np.float32), np.asarray(colour, np.float32)
    shades = np.stack([bg + (c - bg) * s for s in (1 / 3, 2 / 3, 1.0)])
    return np.where(level > 0, k - 1 + level, labels), np.vstack([palette, shades]).astype(np.float32)
