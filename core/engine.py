"""Fast, temporally stable binary-triangle ornament engine.

The stage is a grid of cols x rows square *cells*; every cell is the root of a
binary triangle tree (bintree).  The triangle hierarchy is fixed geometry, so
which triangle a point falls into, at every level, never changes.

    OrnamentTree     per-frame split / merge decisions and animated node colours
                     (cost independent of output resolution)
    NumpyRenderer    reference CPU renderer: precomputes, for every output pixel,
                     its finest triangle and the anti-aliased ornament coverage it
                     would have at every level, so a frame is a few gathers
    OrnamentEngine   1x1 stage + NumpyRenderer behind the original single-square
                     API (used by live.py)

The GPU renderer (align_perform/runtime/gl_renderer.py) draws the same picture
from the same OrnamentTree state without lookup tables.

Temporal stability:
    * EMA on per-node colour histograms (decisions see smoothed statistics)
    * split / merge hysteresis, dominant-colour hysteresis
    * animated split amount: a split cross-fades level L into level L+1
    * animated node colours

Seams:
    * optional conforming (restricted) bintree: splitting a triangle forces its
      base neighbour to split, so neighbouring leaves differ by at most one
      level and every shared edge is a *full* edge on both sides.  Base
      neighbours are matched in stage coordinates, so this holds across cells.
    * optional "cord": a thin background-coloured line between leaves of
      different colour (like the cord sewn over shyrdak seams).

Coordinates:
    Stage coordinates are in cells: x in [0, cols), y in [0, rows), y down.
    Renderers map normalised output coordinates (u right, v down, 0..1 over the
    whole image) to stage coordinates with a 3x3 matrix (affine or homography).

Node indexing:
    With n_sq = cols * rows, level L has n_sq * 2**(L+1) nodes; children of
    (L, i) are (L+1, 2i) and (L+1, 2i+1), so the level-L ancestor of finest
    node f is f >> (max_level - L).  Level-0 node 2c / 2c+1 is the upper-right /
    lower-left half of cell c = y * cols + x.  Per-node state is stored flat,
    level L starting at offset n_sq * (2**(L+1) - 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


POLARITIES = ("alternate", "none", "invert")


@dataclass
class TreeSnapshot:
    """Renderer input: one frame of OrnamentTree state (see OrnamentTree.snapshot)."""

    cols: int
    rows: int
    max_level: int
    amount: np.ndarray               # (n_nodes,) float32
    color: np.ndarray | None         # (n_nodes, 3) float32 BGR, None before the first update
    dom: np.ndarray | None           # (n_nodes,) int32
    version: int = 0
    # Wave rings (core.presence.WaveEmitter): x, y (cells), birth (pipeline clock), strength.
    # Renderers draw them at clock + (time.perf_counter() - wall): the pipeline clock now.
    waves: np.ndarray | None = None  # (n, 4) float64
    clock: float = 0.0               # pipeline clock when the snapshot was taken (s)
    wall: float = 0.0                # time.perf_counter() at that moment

    @property
    def n_nodes(self) -> int:
        return len(self.amount)


# ---------------------------------------------------------------------------
# Ornament tile
# ---------------------------------------------------------------------------

def load_tile(path: str, crop: bool = False) -> np.ndarray:
    """Load a square black-on-white ornament as float32 ink coverage (ink = 1).

    Args:
        path: Ornament image (black ink on white).
        crop: Trim the white margin so the ink touches the tile edges
              (crop stays centred, so the tile's symmetry is preserved).
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load ornament: {path}")
    ink = (gray < 128).astype(np.float32)

    if crop:
        h, w = ink.shape
        ys, xs = np.nonzero(ink)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        half = int(np.ceil(max(np.abs(xs - cx).max(), np.abs(ys - cy).max())))
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        ink = ink[max(0, y0) : y0 + 2 * half + 1, max(0, x0) : x0 + 2 * half + 1]

    # Slight blur so remap sampling is anti-aliased instead of jaggy
    return cv2.GaussianBlur(ink, (0, 0), 0.7)


def inverted(polarity: str, level: int) -> bool:
    """Whether the ornament is drawn inverted at *level*."""
    if polarity == "alternate":
        return level % 2 == 0       # original renderer.py behaviour
    return polarity == "invert"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _descend(points: np.ndarray, levels: int, on_level=None, cols: int = 1, rows: int = 1) -> np.ndarray:
    """Walk every point down the triangle hierarchy.

    Args:
        points:   (P, 2) float64 stage coords in cells (x right, y down).
        levels:   Deepest level to descend to.
        on_level: Optional callback(level, u, v) receiving each point's
                  canonical coordinates inside its level-`level` triangle,
                  p = ra + u * (ha - ra) + v * (hb - ra).
        cols, rows: Stage size in cells.  Points outside are clamped to the
                  nearest cell (their results are meaningless; callers mask them).

    Returns:
        (P,) int64 index of each point's node at `levels`.
    """
    cx = np.clip(np.floor(points[:, 0]), 0, cols - 1)
    cy = np.clip(np.floor(points[:, 1]), 0, rows - 1)
    px, py = points[:, 0] - cx, points[:, 1] - cy
    lower = py > px
    idx = (cy * cols + cx).astype(np.int64) * 2 + lower
    ra = np.where(lower[:, None], (0.0, 1.0), (1.0, 0.0))
    ha = np.where(lower[:, None], (1.0, 1.0), (0.0, 0.0))
    hb = np.where(lower[:, None], (0.0, 0.0), (1.0, 1.0))

    for level in range(levels + 1):
        if on_level is not None:
            e1, e2 = ha - ra, hb - ra
            dx, dy = px - ra[:, 0], py - ra[:, 1]
            det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            u = (dx * e2[:, 1] - dy * e2[:, 0]) / det
            v = (e1[:, 0] * dy - e1[:, 1] * dx) / det
            on_level(level, u, v)
        if level == levels:
            break

        # Split from the right angle to the hypotenuse midpoint;
        # child 2i = (mid, ra, ha), child 2i+1 = (mid, ra, hb).
        mid = (ha + hb) * 0.5
        d = mid - ra
        side_p = d[:, 0] * (py - ra[:, 1]) - d[:, 1] * (px - ra[:, 0])
        side_a = d[:, 0] * (ha[:, 1] - ra[:, 1]) - d[:, 1] * (ha[:, 0] - ra[:, 0])
        first = side_p * side_a >= 0
        idx = idx * 2 + (~first)
        hb = np.where(first[:, None], ha, hb)
        ha = ra
        ra = mid
    return idx


def norm_centres(width: int, height: int) -> np.ndarray:
    """Normalised (0..1) centres of a width x height pixel grid, row-major."""
    xs = (np.arange(width) + 0.5) / width
    ys = (np.arange(height) + 0.5) / height
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def apply_matrix(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 (projective) matrix to (P, 2) points."""
    m = np.asarray(m, dtype=np.float64)
    if np.array_equal(m, np.eye(3)):
        return points
    x, y = points[:, 0], points[:, 1]
    xs = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    ys = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return np.stack([xs / w, ys / w], axis=1)


def root_triangles(cols: int, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ra, ha, hb) corner arrays of the 2 * cols * rows level-0 triangles, in stage coords."""
    gx, gy = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
    gx, gy = gx.ravel(), gy.ravel()
    n = cols * rows
    ra, ha, hb = (np.empty((2 * n, 2)) for _ in range(3))
    # Upper-right half (even index) and lower-left half (odd index) of every cell
    ra[0::2] = np.stack([gx + 1, gy], axis=1)
    ha[0::2] = np.stack([gx, gy], axis=1)
    hb[0::2] = np.stack([gx + 1, gy + 1], axis=1)
    ra[1::2] = np.stack([gx, gy + 1], axis=1)
    ha[1::2] = np.stack([gx + 1, gy + 1], axis=1)
    hb[1::2] = np.stack([gx, gy], axis=1)
    return ra, ha, hb


def _base_neighbours(levels: int, cols: int = 1, rows: int = 1) -> list[np.ndarray]:
    """For every level, the node sharing each node's hypotenuse (-1 on the stage border)."""
    ra, ha, hb = root_triangles(cols, rows)

    result = []
    for level in range(levels + 1):
        # Hypotenuse as an unordered pair of exact (dyadic) endpoints
        a = np.round(ha * (1 << 20)).astype(np.int64)
        b = np.round(hb * (1 << 20)).astype(np.int64)
        swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
        a[swap], b[swap] = b[swap].copy(), a[swap].copy()
        order = np.lexsort((b[:, 1], b[:, 0], a[:, 1], a[:, 0]))
        sa, sb = a[order], b[order]
        same = np.all(sa[1:] == sa[:-1], axis=1) & np.all(sb[1:] == sb[:-1], axis=1)
        i = np.flatnonzero(same)
        nb = np.full(len(ra), -1, dtype=np.int64)
        nb[order[i]] = order[i + 1]
        nb[order[i + 1]] = order[i]
        result.append(nb)

        if level < levels:
            mid = (ha + hb) * 0.5
            new_hb = np.empty((2 * len(ra), 2))
            new_hb[0::2], new_hb[1::2] = ha, hb
            ra, ha, hb = np.repeat(mid, 2, axis=0), np.repeat(ra, 2, axis=0), new_hb
    return result


def _blend(dt: float, tau: float) -> float:
    """Frame-rate independent EMA factor for time constant *tau* seconds."""
    return 1.0 if tau <= 0 else float(1.0 - np.exp(-dt / tau))


# ---------------------------------------------------------------------------
# Tree state
# ---------------------------------------------------------------------------

class OrnamentTree:
    """Adaptive bintree over a cols x rows grid of square cells, with temporal smoothing.

    Usage per frame:
        tree.update(labels, palette_bgr, dt)   # labels: (rows*proc, cols*proc) ints

    Per-node state for renderers:
        amount  (n_nodes,) float32   how far each node is split open (0..1)
        color   (n_nodes, 3) float32 animated BGR colour
        dom     (n_nodes,) int64     dominant palette index
    """

    def __init__(self, cols: int = 1, rows: int = 1, max_level: int = 12, proc_size: int = 128) -> None:
        if cols < 1 or rows < 1:
            raise ValueError("stage needs at least one cell")
        self.cols, self.rows = cols, rows
        self.n_sq = cols * rows
        self.max_level = max_level
        lm = max_level
        self.sizes = np.array([self.n_sq << (l + 1) for l in range(lm + 1)], dtype=np.int64)
        self.offsets = np.array([self.n_sq * ((1 << (l + 1)) - 2) for l in range(lm + 1)], dtype=np.int64)
        self.n_cells = self.n_sq << (lm + 1)
        self.n_nodes = self.n_sq * ((1 << (lm + 2)) - 2)
        self._cells = np.arange(self.n_cells, dtype=np.int64)
        self._node_ar = np.arange(self.n_nodes)
        self._node_level = np.repeat(np.arange(lm + 1), self.sizes)
        self.base_nb = _base_neighbours(lm, cols, rows)

        # Tree decisions
        self.max_depth = 8
        self.min_depth = 0
        self.threshold = 0.75          # split when dominant-colour purity < threshold
        self.hysteresis = 0.08         # ... merge only when purity >= threshold + hysteresis
        self.color_hysteresis = 0.15   # keep dominant unless the best beats it by 15 %
        self.conforming = True

        # Temporal behaviour (seconds)
        self.animate = True
        self.tau_hist = 0.15
        self.tau_color = 0.12
        self.split_time = 0.15         # duration of one level's cross-fade

        # Presence (only when update() gets an activity map)
        self.still_depth = 2           # deepest split level where nothing moves
        self.still_dim = 0.7           # still areas fade this far towards dim_color
        self.dim_color = (181, 217, 253)

        self.set_proc_size(proc_size)
        self.reset()

    # -- configuration -------------------------------------------------------

    @property
    def proc_shape(self) -> tuple[int, int]:
        """(height, width) of the label image update() expects."""
        return self.rows * self.proc_size, self.cols * self.proc_size

    def set_proc_size(self, proc_size: int) -> None:
        """Statistics resolution: pixels per cell side."""
        self.proc_size = proc_size
        h, w = self.proc_shape
        pts = norm_centres(w, h)
        if self.n_sq > 1:
            pts = pts * (self.cols, self.rows)
        self._finest_proc = _descend(pts, self.max_level, cols=self.cols, rows=self.rows)
        self._finest_px = np.bincount(self._finest_proc, minlength=self.n_cells).astype(np.float32)
        self._node_px = self._sum_levels(self._finest_px)
        self._hist = None

    def level_slice(self, level: int) -> slice:
        return slice(int(self.offsets[level]), int(self.offsets[level] + self.sizes[level]))

    def _sum_levels(self, finest: np.ndarray) -> np.ndarray:
        """Per-node sums of a per-finest-cell value, flat node order (level 0 first)."""
        parts = [finest]
        for _ in range(self.max_level):
            parts.append(parts[-1][0::2] + parts[-1][1::2])
        return np.concatenate(parts[::-1])

    def _max_levels(self, finest: np.ndarray) -> np.ndarray:
        parts = [finest]
        for _ in range(self.max_level):
            parts.append(np.maximum(parts[-1][0::2], parts[-1][1::2]))
        return np.concatenate(parts[::-1])

    def _activity_stats(self, activity: np.ndarray, outline: np.ndarray | None):
        """Per node: mean and max activity, and whether the outline's edge cuts it
        (pixels both inside and outside; None without an outline)."""
        px = np.maximum(self._node_px, 1.0)
        a = np.bincount(self._finest_proc, weights=activity.ravel(), minlength=self.n_cells).astype(np.float32)
        mean = self._sum_levels(a) / px
        peak = self._max_levels(a / np.maximum(self._finest_px, 1.0))
        cut = None
        if outline is not None:
            o = np.bincount(self._finest_proc, weights=outline.ravel().astype(np.float32),
                            minlength=self.n_cells).astype(np.float32)
            inside = self._sum_levels(o)
            cut = (inside >= 0.5) & (self._node_px - inside >= 0.5)
        return mean, peak, cut

    def reset(self) -> None:
        self._hist = None
        self._k = None
        self.dom = None
        self.split = np.zeros(self.n_nodes, dtype=bool)
        self.amount = np.zeros(self.n_nodes, dtype=np.float32)
        self.color = None

    @property
    def leaf_count(self) -> int:
        return 2 * self.n_sq + int(self.split.sum())

    # -- per-frame update ----------------------------------------------------

    def update(self, labels: np.ndarray, palette_bgr: np.ndarray, dt: float,
               activity: np.ndarray | None = None, outline: np.ndarray | None = None) -> None:
        """Feed one frame of palette labels, shape proc_shape.

        With *activity* (float 0..1, proc_shape; see core.presence) detail follows
        people: still areas split no deeper than still_depth and fade still_dim of
        the way to dim_color.  *outline* (bool, where someone is this frame) also
        splits triangles that its edge cuts, so silhouettes stay crisp.
        """
        k = len(palette_bgr)
        lm = self.max_level

        counts = np.bincount(
            self._finest_proc * k + labels.ravel(), minlength=self.n_cells * k
        ).astype(np.float32).reshape(self.n_cells, k)

        first = self._hist is None or self._hist.shape != counts.shape or self._k != k
        if first:
            self._hist = counts
            self._k = k
        else:
            self._hist += _blend(dt, self.tau_hist) * (counts - self._hist)

        # Histograms of every node: sum children bottom-up, store level 0 first
        parts = [self._hist]
        for _ in range(lm):
            parts.append(parts[-1][0::2] + parts[-1][1::2])     # (faster than a size-2 reduce)
        hist = np.concatenate(parts[::-1])

        best = hist.argmax(axis=1)
        top = hist[self._node_ar, best]
        total = hist.sum(axis=1)
        purity = np.where(total > 1e-6, top / np.maximum(total, 1e-6), 1.0)
        if activity is not None:
            act_mean, act_peak, cut = self._activity_stats(activity, outline)
            if cut is not None:
                purity = np.where(cut, 0.0, purity)

        if first or self.dom is None:
            dom = best
        else:
            current = hist[self._node_ar, self.dom]
            dom = np.where(current >= top * (1.0 - self.color_hysteresis), self.dom, best)
        self.dom = dom

        # Split / merge with hysteresis
        merge_at = min(self.threshold + self.hysteresis, 1.0)
        split = np.where(purity < self.threshold, True,
                         np.where(purity >= merge_at, False, self.split))
        split[self._node_level < self.min_depth] = True
        split[self._node_level >= min(self.max_depth, lm)] = False
        if activity is not None:
            # Depth allowed per node grows with the most active pixel in it
            still = min(self.still_depth, self.max_depth)
            allowed = np.maximum(still + act_peak * (self.max_depth - still), self.min_depth)
            split[self._node_level >= allowed] = False

        # A node can only be split if its parent is
        for level in range(1, lm + 1):
            split[self.level_slice(level)] &= np.repeat(split[self.level_slice(level - 1)], 2)

        # Conforming closure: split(T) => split(base neighbour of T) and split(parent of T).
        # Implications only point to the same or a coarser level, so one sweep suffices.
        if self.conforming:
            for level in range(lm, -1, -1):
                s = self.level_slice(level)
                cur = split[s]
                nb = self.base_nb[level]
                has = nb >= 0
                forced = cur.copy()
                forced[has] |= cur[nb[has]]
                split[s] = forced
                if level > 0:
                    split[self.level_slice(level - 1)] |= forced.reshape(-1, 2).any(axis=1)
        self.split = split

        # Animate split amounts one level at a time: a child starts opening only
        # once its parent is fully open; a parent starts closing only once its
        # children are fully closed.  So at most two levels ever blend per pixel.
        if first or not self.animate:
            self.amount = split.astype(np.float32)
        else:
            step = dt / max(self.split_time, 1e-3)
            prev = self.amount
            amount = prev.copy()
            for level in range(lm + 1):
                s = self.level_slice(level)
                target = split[s]
                a = prev[s]
                if level > 0:
                    can_open = np.repeat(prev[self.level_slice(level - 1)] >= 1.0, 2)
                else:
                    can_open = np.ones_like(target)
                if level < lm:
                    can_close = prev[self.level_slice(level + 1)].reshape(-1, 2).max(axis=1) <= 0.0
                else:
                    can_close = np.ones_like(target)
                a_new = np.where(target & can_open, np.minimum(a + step, 1.0), a)
                a_new = np.where(~target & can_close, np.maximum(a - step, 0.0), a_new)
                amount[s] = a_new
            self.amount = amount

        # Animate node colours towards their dominant palette colour
        target_col = np.asarray(palette_bgr, dtype=np.float32)[dom]
        if activity is not None and self.still_dim > 0:
            fade = (self.still_dim * (1.0 - act_mean))[:, None]
            target_col += fade * (np.asarray(self.dim_color, dtype=np.float32) - target_col)
        if first or self.color is None or not self.animate or self.color.shape != target_col.shape:
            self.color = target_col
        else:
            self.color += _blend(dt, self.tau_color) * (target_col - self.color)

    # -- per-cell view for renderers ----------------------------------------

    def snapshot(self) -> TreeSnapshot:
        """Copy of the state a renderer needs, safe to hand to another thread."""
        return TreeSnapshot(
            cols=self.cols, rows=self.rows, max_level=self.max_level,
            amount=self.amount.astype(np.float32, copy=True),
            color=None if self.color is None else self.color.astype(np.float32, copy=True),
            dom=None if self.dom is None else self.dom.astype(np.int32, copy=True),
        )

    def visible_levels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per finest cell: (node_a, node_b, t, lvl).

        node_a is the deepest fully-open ancestor's child chain end (the leaf being
        drawn), node_b its child at the next level and t how far node_a is split
        open (> 0 only while that split animates).
        """
        lm = self.max_level
        f = self._cells
        lvl = np.zeros(self.n_cells, dtype=np.int64)
        full = np.ones(self.n_cells, dtype=bool)
        for level in range(lm):
            full &= self.amount[self.offsets[level] + (f >> (lm - level))] >= 1.0
            lvl += full
        lvl_b = np.minimum(lvl + 1, lm)
        node_a = self.offsets[lvl] + (f >> (lm - lvl))
        node_b = self.offsets[lvl_b] + (f >> (lm - lvl_b))
        return node_a, node_b, self.amount[node_a], lvl


# ---------------------------------------------------------------------------
# Reference renderer
# ---------------------------------------------------------------------------

class NumpyRenderer:
    """Renders an OrnamentTree to an image using precomputed per-pixel lookup tables.

    Building the tables takes ~7 s per megapixel, rendering ~50 ms per megapixel,
    so this is for small outputs (live.py, tests); projectors use the GPU renderer.
    """

    def __init__(
        self,
        tree: OrnamentTree,
        size: int | tuple[int, int],
        tile: np.ndarray,
        norm_to_cell: np.ndarray | None = None,
        polarity: str = "alternate",
        supersample: int = 2,
    ) -> None:
        self.tree = tree
        self.tile = tile
        self.polarity = polarity
        self.supersample = supersample
        self.set_output(size, norm_to_cell)

    def set_output(
        self,
        size: int | tuple[int, int],
        norm_to_cell: np.ndarray | None = None,
        polarity: str | None = None,
        tile: np.ndarray | None = None,
    ) -> None:
        """(Re)build the lookup tables.

        Args:
            size:         Output side, or (width, height).
            norm_to_cell: 3x3 matrix from normalised output coords to stage cells.
                          Default: the whole stage.
        """
        if polarity is not None:
            self.polarity = polarity
        if tile is not None:
            self.tile = tile
        t = self.tree
        lm = t.max_level
        w, h = (size, size) if isinstance(size, (int, np.integer)) else size
        self.width, self.height = int(w), int(h)
        if norm_to_cell is None:
            norm_to_cell = np.diag([float(t.cols), float(t.rows), 1.0])
        self.norm_to_cell = np.asarray(norm_to_cell, dtype=np.float64)
        mm = w * h
        self._mm = mm
        self._pix = np.arange(mm, dtype=np.int64)

        pts = apply_matrix(self.norm_to_cell, norm_centres(w, h))
        self._inside = ((pts[:, 0] >= 0) & (pts[:, 0] < t.cols) & (pts[:, 1] >= 0) & (pts[:, 1] < t.rows))
        self._all_inside = bool(self._inside.all())
        self._finest_out = _descend(pts, lm, cols=t.cols, rows=t.rows)

        # Output pixels per cell, for the cord width
        jac = self.norm_to_cell[:2, :2] / self.norm_to_cell[2, 2]
        self._px_per_cell = w / max(np.hypot(jac[0, 0], jac[1, 0]), 1e-9)

        ss = self.supersample
        nw, nh = w * ss, h * ss
        th, tw = self.tile.shape
        coverage = np.empty((lm + 1, mm), dtype=np.uint8)

        def sample(level: int, u: np.ndarray, v: np.ndarray) -> None:
            # Canonical triangle = upper-right half of the tile:
            # right angle -> top-right, ha -> top-left, hb -> bottom-right
            mx = ((1.0 - u) * (tw - 1)).astype(np.float32).reshape(nh, nw)
            my = (v * (th - 1)).astype(np.float32).reshape(nh, nw)
            cov = cv2.remap(self.tile, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if inverted(self.polarity, level):
                cov = 1.0 - cov
            if ss > 1:
                cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_AREA)
            coverage[level] = np.clip(cov * 255 + 0.5, 0, 255).astype(np.uint8).ravel()

        ss_pts = apply_matrix(self.norm_to_cell, norm_centres(nw, nh))
        _descend(ss_pts, lm, sample, cols=t.cols, rows=t.rows)
        self._coverage = coverage.ravel()

    def render(self, background, cord: bool = False, wireframe: bool = False) -> np.ndarray:
        """Render the tree's current state to a (height, width, 3) BGR image.

        Pixels outside the stage are black.
        """
        t = self.tree
        w, h = self.width, self.height
        bg = np.asarray(background, dtype=np.float32)
        if t.color is None:
            image = np.empty((h, w, 3), dtype=np.uint8)
            image[:] = (bg + 0.5).astype(np.uint8)
            image.reshape(-1, 3)[~self._inside] = 0
            return image

        node_a, node_b, t_cell, lvl = t.visible_levels()
        lm = t.max_level
        lvl_b = np.minimum(lvl + 1, lm)
        col_a = t.color[node_a]
        col_b = t.color[node_b]

        # Per output pixel
        fp = self._finest_out
        cov_a = self._coverage[lvl[fp] * self._mm + self._pix].astype(np.float32) * (1 / 255)
        out = bg + cov_a[:, None] * (col_a[fp] - bg)

        t_pix = t_cell[fp]
        moving = np.flatnonzero(t_pix > 0)
        if moving.size:
            fm = fp[moving]
            tm = t_pix[moving][:, None]
            cov_b = self._coverage[lvl_b[fm] * self._mm + moving].astype(np.float32) * (1 / 255)
            out_b = bg + cov_b[:, None] * (col_b[fm] - bg)
            out[moving] = out[moving] * (1 - tm) + out_b * tm

        image = (out + 0.5).astype(np.uint8).reshape(h, w, 3)

        if cord or wireframe:
            visible = np.where(t_cell < 0.5, node_a, node_b)
            key_cell = visible if wireframe else t.dom[visible]
            key = key_cell[fp]
            if not self._all_inside:
                key = np.where(self._inside, key, -1)
            key = key.reshape(h, w)
            edge = np.zeros((h, w), dtype=np.uint8)
            edge[:, 1:] |= key[:, 1:] != key[:, :-1]
            edge[1:, :] |= key[1:, :] != key[:-1, :]
            if cord:
                width = max(1, int(round(self._px_per_cell)) // 256)
                edge = cv2.dilate(edge, np.ones((width, width), np.uint8))
                image[edge > 0] = bg.astype(np.uint8)
            else:
                image[edge > 0] = (0, 0, 0)

        if not self._all_inside:
            image.reshape(-1, 3)[~self._inside] = 0
        return image


# ---------------------------------------------------------------------------
# Single-square engine (live.py)
# ---------------------------------------------------------------------------

def _tree_attr(name: str) -> property:
    return property(lambda self: getattr(self.tree, name),
                    lambda self, value: setattr(self.tree, name, value))


class OrnamentEngine:
    """One square cell + NumpyRenderer behind the original single-square API.

    Usage per frame:
        engine.update(labels, palette_bgr, dt)   # labels: (proc, proc) ints
        image = engine.render()
    """

    max_depth = _tree_attr("max_depth")
    min_depth = _tree_attr("min_depth")
    threshold = _tree_attr("threshold")
    hysteresis = _tree_attr("hysteresis")
    color_hysteresis = _tree_attr("color_hysteresis")
    conforming = _tree_attr("conforming")
    animate = _tree_attr("animate")
    tau_hist = _tree_attr("tau_hist")
    tau_color = _tree_attr("tau_color")
    split_time = _tree_attr("split_time")

    def __init__(
        self,
        tile: np.ndarray,
        out_size: int = 512,
        proc_size: int = 128,
        max_level: int = 12,
        polarity: str = "alternate",
        supersample: int = 2,
        background: tuple[int, int, int] = (181, 217, 253),
    ) -> None:
        self.tree = OrnamentTree(1, 1, max_level, proc_size)
        self.background = np.array(background, dtype=np.float32)
        self.renderer = NumpyRenderer(self.tree, out_size, tile, polarity=polarity, supersample=supersample)

    @property
    def max_level(self) -> int:
        return self.tree.max_level

    @property
    def proc_size(self) -> int:
        return self.tree.proc_size

    @property
    def out_size(self) -> int:
        return self.renderer.width

    @property
    def polarity(self) -> str:
        return self.renderer.polarity

    @property
    def tile(self) -> np.ndarray:
        return self.renderer.tile

    @property
    def leaf_count(self) -> int:
        return self.tree.leaf_count

    def set_proc_size(self, proc_size: int) -> None:
        self.tree.set_proc_size(proc_size)

    def set_output(self, out_size: int, polarity: str | None = None, tile: np.ndarray | None = None) -> None:
        """(Re)build the output-resolution lookup tables (~0.5-2 s)."""
        self.renderer.set_output(out_size, polarity=polarity, tile=tile)

    def reset(self) -> None:
        self.tree.reset()

    def update(self, labels: np.ndarray, palette_bgr: np.ndarray, dt: float) -> None:
        self.tree.update(labels, palette_bgr, dt)

    def render(self, cord: bool = False, wireframe: bool = False) -> np.ndarray:
        return self.renderer.render(self.background, cord=cord, wireframe=wireframe)
