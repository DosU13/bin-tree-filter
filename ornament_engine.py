"""Fast, temporally stable binary-triangle ornament engine.

Why this exists (vs. triangle_tree.py + renderer.py)
-----------------------------------------------------
The triangle hierarchy is *fixed geometry*: which triangle a pixel falls into,
at every level, never changes.  So everything geometric is precomputed once:

    finest_proc[p]   finest-level triangle of every proc pixel    (statistics)
    finest_out[p]    finest-level triangle of every output pixel  (rendering)
    coverage[L, p]   anti-aliased ornament coverage of output pixel p if its
                     level-L triangle were a leaf (polarity baked in)

A frame is then: one bincount -> tiny per-node array ops -> a few per-pixel
gathers.  The cost no longer grows with the number of leaves.

Temporal stability:
    * EMA on per-node colour histograms (decisions see smoothed statistics)
    * split / merge hysteresis, dominant-colour hysteresis
    * animated split amount: a split cross-fades level L into level L+1
    * animated node colours

Seams:
    * optional conforming (restricted) bintree: splitting a triangle forces its
      base neighbour to split, so neighbouring leaves differ by at most one
      level and every shared edge is a *full* edge on both sides.
    * optional "cord": a thin background-coloured line between leaves of
      different colour (like the cord sewn over shyrdak seams).

Node indexing:
    Level L has 2**(L+1) nodes; children of (L, i) are (L+1, 2i) and
    (L+1, 2i+1), so the level-L ancestor of finest node f is f >> (max_level-L).
    Per-node state is stored flat, level L starting at offset 2**(L+1) - 2.
    Node 0 / 1 at level 0 are the upper-right / lower-left halves of the square,
    matching triangle_tree.make_initial_triangles.
"""

from __future__ import annotations

import cv2
import numpy as np


POLARITIES = ("alternate", "none", "invert")


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


# ---------------------------------------------------------------------------
# Geometry precomputation
# ---------------------------------------------------------------------------

def _descend(points: np.ndarray, levels: int, on_level=None) -> np.ndarray:
    """Walk every point down the triangle hierarchy.

    Args:
        points:   (P, 2) float64 coords in the unit square (x right, y down).
        levels:   Deepest level to descend to.
        on_level: Optional callback(level, u, v) receiving each point's
                  canonical coordinates inside its level-`level` triangle,
                  p = ra + u * (ha - ra) + v * (hb - ra).

    Returns:
        (P,) int64 index of each point's node at `levels`.
    """
    px, py = points[:, 0], points[:, 1]
    lower = py > px
    idx = lower.astype(np.int64)
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


def _pixel_centres(size: int) -> np.ndarray:
    c = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(c, c)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _base_neighbours(levels: int) -> list[np.ndarray]:
    """For every level, the node sharing each node's hypotenuse (-1 on the border)."""
    ra = np.array([[1.0, 0.0], [0.0, 1.0]])
    ha = np.array([[0.0, 0.0], [1.0, 1.0]])
    hb = np.array([[1.0, 1.0], [0.0, 0.0]])

    result = []
    for level in range(levels + 1):
        # Hypotenuse as an unordered pair of exact (dyadic) endpoints
        q = np.round(np.concatenate([ha, hb], axis=1) * (1 << 20)).astype(np.int64)
        a = (q[:, 0] << 21) | q[:, 1]
        b = (q[:, 2] << 21) | q[:, 3]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        order = np.lexsort((hi, lo))
        same = (lo[order][1:] == lo[order][:-1]) & (hi[order][1:] == hi[order][:-1])
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
# Engine
# ---------------------------------------------------------------------------

class OrnamentEngine:
    """Adaptive binary-triangle ornament renderer with temporal smoothing.

    Usage per frame:
        engine.update(labels, palette_bgr, dt)   # labels: (proc, proc) ints
        image = engine.render()
    """

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
        self.max_level = max_level
        self.n_cells = 1 << (max_level + 1)
        self.n_nodes = (1 << (max_level + 2)) - 2
        self._offsets = np.array([(1 << (l + 1)) - 2 for l in range(max_level + 1)], dtype=np.int64)
        self._cells = np.arange(self.n_cells, dtype=np.int64)
        self._node_ar = np.arange(self.n_nodes)
        self._node_level = np.repeat(
            np.arange(max_level + 1), [1 << (l + 1) for l in range(max_level + 1)]
        )
        self._base_nb = _base_neighbours(max_level)

        self.tile = tile
        self.polarity = polarity
        self.supersample = supersample
        self.background = np.array(background, dtype=np.float32)

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

        self.set_proc_size(proc_size)
        self.set_output(out_size)
        self.reset()

    # -- configuration -------------------------------------------------------

    def set_proc_size(self, proc_size: int) -> None:
        self.proc_size = proc_size
        self._finest_proc = _descend(_pixel_centres(proc_size), self.max_level)
        self._hist = None

    def set_output(self, out_size: int, polarity: str | None = None, tile: np.ndarray | None = None) -> None:
        """(Re)build the output-resolution lookup tables (~0.5-2 s)."""
        if polarity is not None:
            self.polarity = polarity
        if tile is not None:
            self.tile = tile
        self.out_size = out_size
        mm = out_size * out_size
        self._mm = mm
        self._pix = np.arange(mm, dtype=np.int64)
        self._finest_out = _descend(_pixel_centres(out_size), self.max_level)

        n = out_size * self.supersample
        th, tw = self.tile.shape
        coverage = np.empty((self.max_level + 1, mm), dtype=np.uint8)

        def sample(level: int, u: np.ndarray, v: np.ndarray) -> None:
            # Canonical triangle = upper-right half of the tile:
            # right angle -> top-right, ha -> top-left, hb -> bottom-right
            mx = ((1.0 - u) * (tw - 1)).astype(np.float32).reshape(n, n)
            my = (v * (th - 1)).astype(np.float32).reshape(n, n)
            cov = cv2.remap(self.tile, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if self._inverted(level):
                cov = 1.0 - cov
            if self.supersample > 1:
                cov = cv2.resize(cov, (out_size, out_size), interpolation=cv2.INTER_AREA)
            coverage[level] = np.clip(cov * 255 + 0.5, 0, 255).astype(np.uint8).ravel()

        _descend(_pixel_centres(n), self.max_level, sample)
        self._coverage = coverage.ravel()

    def _inverted(self, level: int) -> bool:
        if self.polarity == "alternate":
            return level % 2 == 0       # original renderer.py behaviour
        return self.polarity == "invert"

    def reset(self) -> None:
        self._hist = None
        self._k = None
        self._dom = None
        self._split = np.zeros(self.n_nodes, dtype=bool)
        self._amount = np.zeros(self.n_nodes, dtype=np.float32)
        self._color = None

    @property
    def leaf_count(self) -> int:
        return 2 + int(self._split.sum())

    # -- per-frame update ----------------------------------------------------

    def update(self, labels: np.ndarray, palette_bgr: np.ndarray, dt: float) -> None:
        """Feed one frame of palette labels (proc_size x proc_size)."""
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
            parts.append(parts[-1].reshape(-1, 2, k).sum(axis=1))
        hist = np.concatenate(parts[::-1])

        best = hist.argmax(axis=1)
        top = hist[self._node_ar, best]
        total = hist.sum(axis=1)
        purity = np.where(total > 1e-6, top / np.maximum(total, 1e-6), 1.0)

        if first or self._dom is None:
            dom = best
        else:
            current = hist[self._node_ar, self._dom]
            dom = np.where(current >= top * (1.0 - self.color_hysteresis), self._dom, best)
        self._dom = dom

        # Split / merge with hysteresis
        merge_at = min(self.threshold + self.hysteresis, 1.0)
        split = np.where(purity < self.threshold, True,
                         np.where(purity >= merge_at, False, self._split))
        split[self._node_level < self.min_depth] = True
        split[self._node_level >= min(self.max_depth, lm)] = False

        # A node can only be split if its parent is
        for level in range(1, lm + 1):
            s = slice(self._offsets[level], self._offsets[level] + (2 << level))
            p = slice(self._offsets[level - 1], self._offsets[level - 1] + (1 << level))
            split[s] &= np.repeat(split[p], 2)

        # Conforming closure: split(T) => split(base neighbour of T) and split(parent of T).
        # Implications only point to the same or a coarser level, so one sweep suffices.
        if self.conforming:
            for level in range(lm, -1, -1):
                s = slice(self._offsets[level], self._offsets[level] + (2 << level))
                cur = split[s]
                nb = self._base_nb[level]
                has = nb >= 0
                forced = cur.copy()
                forced[has] |= cur[nb[has]]
                split[s] = forced
                if level > 0:
                    p = slice(self._offsets[level - 1], self._offsets[level - 1] + (1 << level))
                    split[p] |= forced.reshape(-1, 2).any(axis=1)
        self._split = split

        # Animate split amounts one level at a time: a child starts opening only
        # once its parent is fully open; a parent starts closing only once its
        # children are fully closed.  So at most two levels ever blend per pixel.
        if first or not self.animate:
            self._amount = split.astype(np.float32)
        else:
            step = dt / max(self.split_time, 1e-3)
            prev = self._amount
            amount = prev.copy()
            for level in range(lm + 1):
                s = slice(self._offsets[level], self._offsets[level] + (2 << level))
                target = split[s]
                a = prev[s]
                if level > 0:
                    p = slice(self._offsets[level - 1], self._offsets[level - 1] + (1 << level))
                    can_open = np.repeat(prev[p] >= 1.0, 2)
                else:
                    can_open = np.ones_like(target)
                if level < lm:
                    c = slice(self._offsets[level + 1], self._offsets[level + 1] + (4 << level))
                    can_close = prev[c].reshape(-1, 2).max(axis=1) <= 0.0
                else:
                    can_close = np.ones_like(target)
                a_new = np.where(target & can_open, np.minimum(a + step, 1.0), a)
                a_new = np.where(~target & can_close, np.maximum(a - step, 0.0), a_new)
                amount[s] = a_new
            self._amount = amount

        # Animate node colours towards their dominant palette colour
        target_col = np.asarray(palette_bgr, dtype=np.float32)[dom]
        if first or self._color is None or not self.animate:
            self._color = target_col
        else:
            self._color += _blend(dt, self.tau_color) * (target_col - self._color)

    # -- rendering -----------------------------------------------------------

    def render(self, cord: bool = False, wireframe: bool = False) -> np.ndarray:
        """Render the current state to an (out_size, out_size, 3) BGR image."""
        lm = self.max_level
        f = self._cells
        amount = self._amount

        # Per finest cell: primary level = depth of the fully-open ancestor chain
        lvl = np.zeros(self.n_cells, dtype=np.int64)
        full = np.ones(self.n_cells, dtype=bool)
        for level in range(lm):
            full &= amount[self._offsets[level] + (f >> (lm - level))] >= 1.0
            lvl += full
        lvl_b = np.minimum(lvl + 1, lm)
        node_a = self._offsets[lvl] + (f >> (lm - lvl))
        node_b = self._offsets[lvl_b] + (f >> (lm - lvl_b))
        t_cell = amount[node_a]            # > 0 only while level lvl is splitting
        col_a = self._color[node_a]
        col_b = self._color[node_b]

        # Per output pixel
        fp = self._finest_out
        bg = self.background
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

        size = self.out_size
        image = (out + 0.5).astype(np.uint8).reshape(size, size, 3)

        if cord or wireframe:
            visible = np.where(t_cell < 0.5, node_a, node_b)
            key_cell = visible if wireframe else self._dom[visible]
            key = key_cell[fp].reshape(size, size)
            edge = np.zeros((size, size), dtype=np.uint8)
            edge[:, 1:] |= key[:, 1:] != key[:, :-1]
            edge[1:, :] |= key[1:, :] != key[:-1, :]
            if cord:
                width = max(1, size // 256)
                edge = cv2.dilate(edge, np.ones((width, width), np.uint8))
                image[edge > 0] = bg.astype(np.uint8)
            else:
                image[edge > 0] = (0, 0, 0)
        return image
