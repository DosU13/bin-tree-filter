"""What a renderer needs to draw one output: RenderView, edge blending, and their scene builders.

Renderer-agnostic (used by the GPU renderer, the NumPy fallback and tools).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core import geometry as geo
from core.filters import hex_to_bgr


@dataclass
class BlendSpec:
    """Edge blending between overlapping projectors.

    quads:   (n, 4, 2) every enabled projector's area in stage cells (TL, TR, BR, BL;
             any winding).  Edge k runs from corner k to corner k + 1.
    feather: (n,) ramp width per projector, in stage cells.
    index:   which quad is the projector being drawn.
    gamma:   projector gamma; light adds linearly, so weights go through ^(1/gamma).
    edges:   which edges a projector fades towards: only those running inside
             another projector, so outer borders stay at full brightness.
    """

    quads: np.ndarray
    feather: np.ndarray
    index: int
    gamma: float = 2.2
    edges: np.ndarray | None = None      # (n, 4) bool: edges that fade (default: all)


@dataclass
class WaveStyle:
    """How wave rings look (core.presence.WaveEmitter starts them; GPU renderer only)."""

    speed: float = 0.6                 # cells / s
    width: float = 0.12                # cells
    range: float = 1.5                 # cells: a ring has faded out by here
    depth: int = 3                     # triangles under a ring's crest split this many more times
    glow: float = 0.6                  # the crest's ink tints this far towards color
    color: tuple = (45, 35, 180)       # BGR 0..255


@dataclass
class RenderView:
    """Where and how to draw.

    norm_to_cell: 3x3 matrix from normalised output coords (u right, v down,
                  0..1 over the whole output) to stage cells.
    waves / clock: wave rings come with the TreeSnapshot; they are drawn at
                  pipeline time *clock*, or (None) at the snapshot's clock plus the
                  wall time since it was taken, so rings move smoothly between frames.
    """

    width: int
    height: int
    norm_to_cell: np.ndarray
    background: tuple = (181, 217, 253)    # BGR 0..255
    polarity: str = "alternate"
    cord: bool = False
    wireframe: bool = False
    supersample: int = 2
    mip: bool = True
    gain: float = 1.0
    blend: BlendSpec | None = None
    outside: tuple = (0, 0, 0)             # BGR outside the stage
    flat: tuple | None = None              # BGR: draw this flat colour instead (test pattern)
    waves: WaveStyle | None = None         # None: don't draw wave rings
    clock: float | None = None

    @property
    def px_per_cell(self) -> float:
        m = np.asarray(self.norm_to_cell, dtype=np.float64)
        jac = m[:2, :2] / m[2, 2]
        return self.width / max(float(np.hypot(jac[0, 0], jac[1, 0])), 1e-9)

    @property
    def cord_width(self) -> int:
        return max(1, int(round(self.px_per_cell)) // 256)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def render_background(params: dict) -> tuple[int, int, int]:
    """Colour behind the ornament ink for the current mode (BGR)."""
    return hex_to_bgr(params["depth_bg"] if params.get("mode") == "depth" else params["background"])


# Params that change how a frame is drawn (everything else only changes the tree)
STYLE_KEYS = ("mode", "background", "depth_bg", "polarity", "cord", "wireframe", "waves", "wave_speed",
              "wave_width", "wave_range", "wave_depth", "wave_glow", "wave_color")


def style_key(params: dict) -> tuple:
    """Changes whenever a view built from *params* would look different."""
    return tuple(params.get(k) for k in STYLE_KEYS)


def wave_style(params: dict) -> WaveStyle | None:
    if not params.get("waves"):
        return None
    return WaveStyle(speed=params["wave_speed"], width=params["wave_width"], range=params["wave_range"],
                     depth=params["wave_depth"], glow=params["wave_glow"], color=hex_to_bgr(params["wave_color"]))


def _style(params: dict) -> dict:
    return dict(background=render_background(params), polarity=params["polarity"],
                cord=bool(params["cord"]), wireframe=bool(params["wireframe"]), waves=wave_style(params))


def stage_view(scene, width: int, height: int | None = None, params: dict | None = None,
               margin: float = 0.0) -> RenderView:
    """The whole stage (plus *margin* world px around it), e.g. for previews."""
    st = scene.stage
    if height is None:
        height = max(1, round(width * (st.height + 2 * margin) / (st.width + 2 * margin)))
    m = st.world_to_cell() @ geo.rect_matrix(st.x - margin, st.y - margin,
                                              st.width + 2 * margin, st.height + 2 * margin)
    return RenderView(width, height, m, **_style(params or scene.params))


def world_view(scene, x: float, y: float, w: float, h: float, width: int, height: int,
               params: dict | None = None) -> RenderView:
    """An arbitrary world rectangle (canvas previews)."""
    m = scene.stage.world_to_cell() @ geo.rect_matrix(x, y, w, h)
    return RenderView(width, height, m, **_style(params or scene.params))


def projector_quads_cells(scene) -> tuple[list[str], np.ndarray]:
    """Ids and (n, 4, 2) stage-cell quads of all enabled projectors."""
    w2c = scene.stage.world_to_cell()
    projs = scene.enabled_projectors()
    quads = np.array([geo.apply(w2c, p.transform.quad()) for p in projs]).reshape(-1, 4, 2)
    return [p.id for p in projs], quads


def _bbox_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Narrowest side of the bounding-box intersection of two quads (0 if apart)."""
    lo = np.maximum(a.min(axis=0), b.min(axis=0))
    hi = np.minimum(a.max(axis=0), b.max(axis=0))
    size = hi - lo
    return float(size.min()) if np.all(size > 0) else 0.0


def interior_edges(quads: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(n, 4) bool: edge k of quad i runs (partly) inside another quad."""
    n = len(quads)
    out = np.zeros((n, 4), dtype=bool)
    ts = np.linspace(0.1, 0.9, 9)
    for i in range(n):
        for k in range(4):
            a, b = quads[i][k], quads[i][(k + 1) % 4]
            pts = a[None] + (b - a)[None] * ts[:, None]
            out[i, k] = any(np.any(geo.quad_distance(pts, quads[j]) > eps) for j in range(n) if j != i)
    return out


def blend_spec(scene, projector) -> BlendSpec | None:
    """Edge-blend settings for *projector*, or None when it overlaps no other projector."""
    ids, quads = projector_quads_cells(scene)
    if projector.id not in ids or len(ids) < 2:
        return None
    quads = np.array([geo.orient_quad(q) for q in quads])
    cs = scene.stage.cell_size
    feather = np.zeros(len(ids))
    any_overlap = False
    for i, pid in enumerate(ids):
        p = scene.projector(pid)
        overlaps = [_bbox_overlap(quads[i], quads[j]) for j in range(len(ids)) if j != i]
        widest = max(overlaps, default=0.0)
        any_overlap |= widest > 0 and pid == projector.id
        if p.blend.feather is not None:
            feather[i] = max(p.blend.feather / cs, 1e-6)
        else:
            feather[i] = max(widest, 1e-6)
    if not any_overlap:
        return None
    return BlendSpec(quads, feather, ids.index(projector.id), gamma=projector.blend.gamma,
                     edges=interior_edges(quads))


def projector_view(scene, projector, width: int | None = None, height: int | None = None,
                   params: dict | None = None, blend: bool = True) -> RenderView:
    """RenderView for one projector's output (defaults to its native resolution)."""
    w = width or projector.output.width
    h = height or projector.output.height
    m = scene.stage.world_to_cell() @ projector.transform.norm_to_world()
    view = RenderView(w, h, m, gain=projector.gain, **_style(params or scene.params))
    if blend:
        view.blend = blend_spec(scene, projector)
    return view


def blend_alpha(points_cells: np.ndarray, spec: BlendSpec) -> np.ndarray:
    """CPU version of the shader's edge-blend factor (already ^(1/gamma)) for stage points."""
    def smoothstep(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    pts = np.asarray(points_cells, dtype=np.float64).reshape(-1, 2)
    wsum = np.zeros(len(pts))
    wself = np.zeros(len(pts))
    edges = spec.edges if spec.edges is not None else np.ones((len(spec.quads), 4), bool)
    for q in range(len(spec.quads)):
        quad = np.asarray(spec.quads[q], dtype=np.float64)
        inside = geo.quad_distance(pts, quad) > 0
        w = np.ones(len(pts))
        for k in range(4):
            if edges[q, k]:
                a, b = quad[k], quad[(k + 1) % 4]
                e = b - a
                n = np.array([-e[1], e[0]]) / max(np.hypot(*e), 1e-12)
                w *= smoothstep(0.0, max(spec.feather[q], 1e-6), (pts - a) @ n)
        w = np.where(inside, w + 1e-4, 0.0)
        wsum += w
        if q == spec.index:
            wself = w
    alpha = np.where(wsum > 0, wself / np.maximum(wsum, 1e-12), 1.0)
    return alpha ** (1.0 / spec.gamma)
