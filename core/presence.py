"""Who is on stage: moving people found against a learned picture of the empty stage.

    PresenceTracker.update(image, covered, dt, ...) -> Presence
        activity  float32 0..1 at stage proc resolution: 1 where someone is now,
                  fading trails where someone has just been
        mask      bool: where someone is this frame
        people    blobs tracked over frames (stable ids, positions in stage cells)

    WaveEmitter.update(people, clock, ...) -> (n, 4) wave rings for the renderers

Background model: a running average of the empty stage.  Only areas where
nobody is are learned (selective update), so people don't fade into it, but
whatever stays perfectly still for about `absorb` seconds does (a moved chair,
or the "ghost" of someone who stood there while the background was learned).
Each frame is first matched to the background's overall brightness and colour,
so camera auto-exposure and white-balance drifts don't light up the stage; if
most of the stage still differs for a couple of seconds (lights switched, a
camera bumped) the background is relearned.

Cost at the default 128 px per cell: about 0.5 ms per cell per frame (one CPU core).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

BG_ADAPT_SECONDS = 4.0      # empty areas follow slow lighting changes with this time constant
LEARN_SECONDS = 0.5         # relearn(): average the empty stage over this long
RELEARN_FRACTION = 0.6      # this share of the stage "moving" ...
RELEARN_SECONDS = 2.0       # ... for this long means the picture changed: relearn
TRACK_RADIUS = 0.6          # cells: a blob continues the nearest track this close
TRACK_GRACE = 0.5           # s: a track that loses its blob survives this long
ARRIVE_SECONDS = 0.2        # s: a blob must last this long before it counts as someone arriving
UPRIGHT_GAP = 0.15          # cells: blobs stacked closer than this are one person
MAX_WAVES = 16              # rings drawn at once (the shader's array size)
PULSE_STRENGTH = 0.8        # repeat rings are a little weaker than the arrival ring


@dataclass
class Person:
    """One tracked blob.  Positions are in stage cells (x right, y down)."""

    id: int
    x: float                 # centre of the outline
    y: float
    foot_x: float            # bottom centre of the outline
    foot_y: float
    area: float              # cells²
    age: float = 0.0         # seconds tracked
    missing: float = 0.0     # seconds since its blob was last seen (0 = seen this frame)

    def origin(self, where: str) -> tuple[float, float]:
        """Where its wave rings start: "feet" or "body"."""
        return (self.foot_x, self.foot_y) if where == "feet" else (self.x, self.y)


@dataclass
class Presence:
    activity: np.ndarray     # float32 0..1
    mask: np.ndarray         # bool
    people: list[Person]     # seen this frame or lost less than TRACK_GRACE ago
    learning: bool = False   # the background is being (re)learned this frame

    @property
    def count(self) -> int:
        return sum(1 for p in self.people if p.missing == 0)


def _odd(n: float) -> int:
    return max(3, int(round(n)) | 1)


def _merge_upright(blobs: list[list], gap: float) -> list[list]:
    """Join blobs stacked one above the other (a torso and legs split where the
    clothes match the wall).  Blob: [x0, y0, x1, y1, area, sum x, sum y] in px."""
    blobs = [list(b) for b in blobs]
    merged = True
    while merged:
        merged = False
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                a, b = blobs[i], blobs[j]
                overlap = min(a[2], b[2]) - max(a[0], b[0])
                vgap = max(a[1], b[1]) - min(a[3], b[3])
                if overlap >= 0.5 * min(a[2] - a[0], b[2] - b[0]) and vgap <= gap:
                    blobs[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                                a[4] + b[4], a[5] + b[5], a[6] + b[6]]
                    del blobs[j]
                    merged = True
                    break
            if merged:
                break
    return blobs


class PresenceTracker:
    """Motion mask, trails and people for the stitched stage image (see module docstring)."""

    def __init__(self) -> None:
        self.background: np.ndarray | None = None      # float32 BGR, stage proc resolution
        self.activity: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.tracks: list[Person] = []
        self._busy: np.ndarray | None = None              # last mask dilated (uint8)
        self._next_id = 1
        self._learn_left = 0.0
        self._learn_frames = 0
        self._changed_for = 0.0

    def relearn(self, seconds: float = LEARN_SECONDS) -> None:
        """Take the next *seconds* of frames as the empty stage (call it with nobody on stage)."""
        self._learn_left = max(seconds, 1e-6)
        self._learn_frames = 0
        self._changed_for = 0.0

    def reset(self) -> None:
        self.background = None
        self.activity = None
        self.mask = None
        self._busy = None
        self.tracks = []

    @property
    def learning(self) -> bool:
        return self._learn_left > 0

    # -- per frame -----------------------------------------------------------

    def update(self, image: np.ndarray, covered: np.ndarray, dt: float, threshold: float = 24,
               absorb: float = 60.0, trail: float = 1.5, min_size: float = 1.0,
               cell_px: int = 128) -> Presence:
        """One frame.

        Args:
            image:     stitched stage, BGR uint8 (or float), proc resolution
            covered:   bool, pixels some camera sees
            threshold: colour difference (0-255, largest channel) that counts as a person
            absorb:    time constant (s) for still things to become background; 0 = never
            trail:     time constant (s) of the activity trails
            min_size:  smallest blob kept, % of one cell's area
            cell_px:   proc pixels per cell side
        """
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        if self.background is None or self.background.shape != img.shape:
            self.background = img.copy()
            self.activity = np.zeros((h, w), np.float32)
            self.tracks = []
            self.relearn()
        cov = covered.astype(np.uint8)
        small = np.ones((_odd(cell_px / 40),) * 2, np.uint8)
        big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cell_px / 20),) * 2)

        learning = self.learning
        if learning:
            # Running mean over the learning period (the first frame replaces the old picture)
            self._learn_frames += 1
            cv2.accumulateWeighted(img, self.background, 1.0 / self._learn_frames)
            self._learn_left -= dt
            mask = np.zeros((h, w), bool)
            blobs = []
        else:
            mask, blobs = self._detect(img, cov, threshold, min_size, cell_px, small, big)
            covered_px = max(cv2.countNonZero(cov), 1)
            if np.count_nonzero(mask) > RELEARN_FRACTION * covered_px:
                self._changed_for += dt
                if self._changed_for >= RELEARN_SECONDS:
                    log.info("presence: most of the stage changed for %.0f s; relearning the background",
                             RELEARN_SECONDS)
                    self.relearn()
            else:
                self._changed_for = 0.0
            self._learn_background(img, cov, mask, dt, absorb, big)

        act = mask.astype(np.float32)
        if trail > 0:
            np.maximum(act, self.activity * math.exp(-dt / trail), out=act)
        act[cov == 0] = 0.0
        self.activity = act
        self.mask = mask
        self._track(blobs, dt, cell_px)
        return Presence(act, mask, list(self.tracks), learning)

    def _detect(self, img, cov, threshold, min_size, cell_px, small, big):
        bg = self.background
        # Exposure / white balance: match the background's mean colour over areas
        # expected to be empty (last frame's people and their margin left out)
        ref = cov
        if self._busy is not None and self._busy.shape == cov.shape:
            ref = cov & (self._busy == 0)
        gain = (1.0, 1.0, 1.0, 0.0)
        if cv2.countNonZero(ref) > 0.05 * ref.size:
            m_img, m_bg = cv2.mean(img, ref), cv2.mean(bg, ref)
            gain = tuple(min(2.0, max(0.5, a / max(b, 1.0))) for a, b in zip(m_img[:3], m_bg[:3])) + (0.0,)
        b, g, r = cv2.split(cv2.absdiff(img, cv2.multiply(bg, gain)))
        raw = cv2.compare(cv2.max(cv2.max(b, g), r), float(threshold), cv2.CMP_GT)    # 0 / 255
        raw = cv2.bitwise_and(raw, raw, mask=cov)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, small)      # speckle noise
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, big)       # join a person's parts
        n, labels, stats, cents = cv2.connectedComponentsWithStats(raw, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_size / 100.0 * cell_px * cell_px
        keep[0] = False
        mask = raw > 0 if keep[1:].all() else keep[labels]
        blobs = []
        for i in np.flatnonzero(keep):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            blobs.append([x, y, x + w, y + h, area, cents[i][0] * area, cents[i][1] * area])
        return mask, _merge_upright(blobs, UPRIGHT_GAP * cell_px)

    def _learn_background(self, img, cov, mask, dt, absorb, big) -> None:
        busy = cv2.dilate(mask.astype(np.uint8), big)          # people plus a margin
        self._busy = busy
        empty = cov & (busy == 0)
        cv2.accumulateWeighted(img, self.background, 1.0 - math.exp(-dt / BG_ADAPT_SECONDS), mask=empty)
        if absorb > 0:
            cv2.accumulateWeighted(img, self.background, 1.0 - math.exp(-dt / absorb), mask=cov & busy)

    def _track(self, blobs, dt: float, cell_px: int) -> None:
        found = []
        for x0, y0, x1, y1, area, sx, sy in blobs:
            found.append(Person(0, sx / area / cell_px, sy / area / cell_px, (x0 + x1) / 2 / cell_px,
                                y1 / cell_px, area / (cell_px * cell_px)))
        pairs = sorted((math.hypot(t.x - b.x, t.y - b.y), ti, bi)
                       for ti, t in enumerate(self.tracks) for bi, b in enumerate(found))
        used_t, used_b = set(), set()
        for d, ti, bi in pairs:
            if d > TRACK_RADIUS:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            t, b = self.tracks[ti], found[bi]
            t.x, t.y, t.foot_x, t.foot_y, t.area = b.x, b.y, b.foot_x, b.foot_y, b.area
            t.missing = 0.0
        for ti, t in enumerate(self.tracks):
            t.age += dt
            if ti not in used_t:
                t.missing += dt
        self.tracks = [t for t in self.tracks if t.missing <= TRACK_GRACE]
        for bi, b in enumerate(found):
            if bi not in used_b:
                b.id = self._next_id
                self._next_id += 1
                self.tracks.append(b)


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------

class WaveEmitter:
    """Wave rings started by people: one when someone arrives, then one every
    *interval* seconds while they stay.  Rings live for *lifetime* seconds."""

    def __init__(self) -> None:
        self.waves: list[tuple[float, float, float, float]] = []   # x, y (cells), birth (clock), strength
        self._last: dict[int, float] = {}

    def update(self, people: list[Person], clock: float, interval: float, origin: str,
               lifetime: float) -> np.ndarray:
        alive = set()
        for p in people:
            alive.add(p.id)
            if p.missing > 0 or p.age < ARRIVE_SECONDS:
                continue
            last = self._last.get(p.id)
            if last is None or (interval > 0 and clock - last >= interval):
                x, y = p.origin(origin)
                self.waves.append((x, y, clock, 1.0 if last is None else PULSE_STRENGTH))
                self._last[p.id] = clock
        self._last = {k: v for k, v in self._last.items() if k in alive}
        self.waves = [w for w in self.waves if clock - w[2] <= lifetime][-MAX_WAVES:]
        return self.array()

    def array(self) -> np.ndarray:
        return np.array(self.waves, dtype=np.float64).reshape(-1, 4)

    def clear(self) -> None:
        self.waves = []
        self._last = {}


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def draw_presence(image: np.ndarray, activity: np.ndarray | None, people: list[Person],
                  cell_px: int) -> np.ndarray:
    """Debug view: the stage dimmed, activity in red, each person's centre and feet marked."""
    out = (image.astype(np.float32) * 0.45)
    if activity is not None and activity.shape == image.shape[:2]:
        out[:, :, 2] = np.maximum(out[:, :, 2], activity * 255)
        out[:, :, 1] += activity * 40
    out = np.clip(out, 0, 255).astype(np.uint8)
    for p in people:
        colour = (255, 255, 255) if p.missing == 0 else (128, 128, 128)
        c = (int(p.x * cell_px), int(p.y * cell_px))
        f = (int(p.foot_x * cell_px), int(p.foot_y * cell_px) - 1)
        cv2.circle(out, c, 3, colour, 1, cv2.LINE_AA)
        cv2.drawMarker(out, f, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
        cv2.putText(out, str(p.id), (c[0] + 5, c[1] - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, colour, 1, cv2.LINE_AA)
    return out
