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
             any winding).
    feather: (n,) ramp width per projector, in stage cells.
    index:   which quad is the projector being drawn.
    gamma:   projector gamma; light adds linearly, so weights go through ^(1/gamma).
    """

    quads: np.ndarray
    feather: np.ndarray
    index: int
    gamma: float = 2.2


@dataclass
class RenderView:
    """Where and how to draw.

    norm_to_cell: 3x3 matrix from normalised output coords (u right, v down,
                  0..1 over the whole output) to stage cells.
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


def _style(params: dict) -> dict:
    return dict(background=render_background(params), polarity=params["polarity"],
                cord=bool(params["cord"]), wireframe=bool(params["wireframe"]))


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


def blend_spec(scene, projector) -> BlendSpec | None:
    """Edge-blend settings for *projector*, or None when it overlaps no other projector."""
    ids, quads = projector_quads_cells(scene)
    if projector.id not in ids or len(ids) < 2:
        return None
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
    return BlendSpec(quads, feather, ids.index(projector.id), gamma=projector.blend.gamma)


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

    wsum = np.zeros(len(points_cells))
    wself = np.zeros(len(points_cells))
    for q in range(len(spec.quads)):
        d = geo.quad_distance(points_cells, spec.quads[q])
        w = np.where(d <= 0, 0.0, smoothstep(0.0, max(spec.feather[q], 1e-6), d) + 1e-4)
        wsum += w
        if q == spec.index:
            wself = w
    alpha = np.where(wsum > 0, wself / np.maximum(wsum, 1e-12), 1.0)
    return alpha ** (1.0 / spec.gamma)
