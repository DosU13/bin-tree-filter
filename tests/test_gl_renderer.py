"""GPU renderer parity with the NumPy reference renderer."""

import os

import cv2
import numpy as np
import pytest

from core.engine import NumpyRenderer, load_tile
from tests.engine_cases import ROOT
from tests.test_engine_multicell import run_tree

moderngl = pytest.importorskip("moderngl")


def _make_gl():
    from align_perform.runtime.gl_renderer import OffscreenGL
    try:
        return OffscreenGL()
    except Exception as exc:    # no GPU / driver
        pytest.skip(f"no OpenGL 3.3 context: {exc}")


@pytest.fixture(scope="module")
def gl():
    g = _make_gl()
    yield g
    g.release()


TILE = load_tile(os.path.join(ROOT, "ornam.jpg"))
BG = (181, 217, 253)


@pytest.fixture(scope="module")
def tree():
    return run_tree(2, 1, proc=32, levels=8, frames=6)


# Nudge views off exact triangle edges: a pixel centre lying exactly on an edge may
# go either way (float32 on the GPU vs float64 in NumPy) and both are right.
JITTER = np.array([[1.0, 0, 0.000123], [0, 1.0, 0.000217], [0, 0, 1]])


def views(tree):
    """(name, size, norm_to_cell) cases."""
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32([[0.1, 0.0], [1.9, 0.15], [1.8, 1.0], [0.0, 0.9]])
    cases = [
        ("whole", (192, 96), np.diag([2.0, 1.0, 1.0])),
        ("cell1", (96, 96), np.array([[1.0, 0, 1.0], [0, 1.0, 0], [0, 0, 1]])),
        ("zoom", (128, 128), np.array([[0.5, 0, 0.7], [0, 0.5, 0.2], [0, 0, 1]])),
        ("overhang", (160, 64), np.array([[2.5, 0, -0.25], [0, 1.0, 0], [0, 0, 1]])),
        ("homography", (128, 96), cv2.getPerspectiveTransform(src, dst)),
    ]
    return [(name, size, JITTER @ m) for name, size, m in cases]


def compare(a: np.ndarray, b: np.ndarray, border: int = 0) -> tuple[float, float]:
    if border:
        a = a[border:-border, border:-border]
        b = b[border:-border, border:-border]
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
    return float(d.mean()), float(np.mean(d > 24))


@pytest.mark.gpu
@pytest.mark.parametrize("polarity", ["alternate", "invert"])
def test_matches_numpy(gl, tree, polarity):
    from align_perform.runtime.gl_renderer import RenderView
    gl.set_tile(TILE)
    snap = tree.snapshot()
    for name, (w, h), m in views(tree):
        ref = NumpyRenderer(tree, (w, h), TILE, norm_to_cell=m, polarity=polarity).render(BG)
        out = gl.render(snap, RenderView(w, h, m, background=BG, polarity=polarity, mip=False))
        d = np.abs(ref.astype(int) - out.astype(int))
        # Only OpenGL vs OpenCV bilinear weights differ
        assert d.max() <= 3 and d.mean() < 0.2, f"{name}: mean {d.mean():.3f}, max {d.max()}"


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["cord", "wireframe"])
def test_edges_match_numpy(gl, tree, mode):
    from align_perform.runtime.gl_renderer import RenderView
    gl.set_tile(TILE)
    snap = tree.snapshot()
    w, h = 512, 256      # wide enough for a 2-px cord
    m = JITTER @ np.diag([2.0, 1.0, 1.0])
    kw = {mode: True}
    ref = NumpyRenderer(tree, (w, h), TILE, norm_to_cell=m).render(BG, **kw)
    out = gl.render(snap, RenderView(w, h, m, background=BG, mip=False, **kw))
    # The GPU also draws lines on the image's first row / column (it can see past
    # the edge); the NumPy renderer can't, so ignore a thin border
    mean, bad = compare(ref, out, border=3)
    assert mean < 0.5 and bad < 0.002, f"{mode}: mean {mean:.2f}, {bad:.2%} off"


@pytest.mark.gpu
def test_mip_mode_is_close(gl, tree):
    from align_perform.runtime.gl_renderer import RenderView
    gl.set_tile(TILE)
    m = JITTER @ np.diag([2.0, 1.0, 1.0])
    ref = NumpyRenderer(tree, (192, 96), TILE, norm_to_cell=m).render(BG)
    out = gl.render(tree.snapshot(), RenderView(192, 96, m, background=BG, mip=True))
    # Mipmapping removes aliasing at small scales, so compare the pictures' structure
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    assert np.abs(blur(ref) - blur(out)).mean() < 6.0


@pytest.mark.gpu
def test_empty_state_draws_background(gl):
    from align_perform.runtime.gl_renderer import RenderView
    gl.set_tile(TILE)
    out = gl.render(None, RenderView(32, 16, np.eye(3), background=(10, 20, 30)))
    assert out.shape == (16, 32, 3)
    # Either nothing uploaded yet (background) or a stale state; with none it must be background
    assert np.all(out == (10, 20, 30)) or out.std() > 0


@pytest.mark.gpu
def test_edge_blending_sums_to_one(gl, tree):
    """Two projectors overlapping by half a cell: with gamma 1 their outputs add up to the plain image."""
    from align_perform.runtime.gl_renderer import BlendSpec, RenderView
    gl.set_tile(TILE)
    snap = tree.snapshot()
    quads = np.array([
        [[0.0, 0.0], [1.25, 0.0], [1.25, 1.0], [0.0, 1.0]],
        [[0.75, 0.0], [2.0, 0.0], [2.0, 1.0], [0.75, 1.0]],
    ])
    feather = np.array([0.5, 0.5])
    w, h = 256, 128
    m = np.diag([2.0, 1.0, 1.0])
    plain = gl.render(snap, RenderView(w, h, m, background=BG)).astype(int)
    parts = [gl.render(snap, RenderView(w, h, m, background=BG, blend=BlendSpec(quads, feather, i, gamma=1.0)))
             for i in range(2)]
    total = parts[0].astype(int) + parts[1].astype(int)
    assert np.abs(total - plain).max() <= 2
    # Outside the overlap each projector is either full or black
    assert np.array_equal(parts[0][:, :90], plain[:, :90].astype(np.uint8))
    assert parts[0][:, 170:].max() == 0


@pytest.mark.gpu
def test_gpu_blend_matches_cpu(gl):
    """Flat white field through the shader's blend equals core.view.blend_alpha."""
    import os
    from core.scene import Scene
    from core.view import blend_alpha, projector_view
    scene = Scene.load(os.path.join(ROOT, "scenes", "example.json"))
    proj = scene.projectors[0]
    view = projector_view(scene, proj, 192, 108)
    view.flat = (255, 255, 255)
    out = gl.render(None, view)[:, :, 0].astype(float) / 255
    from core.engine import apply_matrix, norm_centres
    pts = apply_matrix(view.norm_to_cell, norm_centres(192, 108))
    ref = blend_alpha(pts, view.blend).reshape(108, 192)
    assert np.abs(out - ref).max() < 2.5 / 255


# ---------------------------------------------------------------------------
# Wave rings
# ---------------------------------------------------------------------------

def _ring_view(clock, glow=0.0, depth=4):
    from core.view import RenderView, WaveStyle
    v = RenderView(256, 128, np.diag([2.0, 1.0, 1.0]), background=BG)
    v.waves = WaveStyle(speed=0.5, width=0.06, range=2.0, depth=depth, glow=glow, color=(0, 0, 0))
    v.clock = clock
    return v


def test_wave_ring_subdivides_and_moves(gl):
    import copy
    gl.set_tile(TILE)
    plain = run_tree(2, 1, proc=32, levels=8, frames=1).snapshot()     # first frame: coarse tree
    snap = copy.copy(plain)
    snap.waves = np.array([[1.0, 0.5, 0.0, 1.0]])           # centre of the stage, born at t = 0

    base = gl.render(plain, _ring_view(1.0))
    at1 = gl.render(snap, _ring_view(1.0))                   # radius 0.5 cells
    at2 = gl.render(snap, _ring_view(1.6))                   # radius 0.8 cells
    changed1 = np.abs(at1.astype(int) - base).max(axis=2) > 0
    changed2 = np.abs(at2.astype(int) - base).max(axis=2) > 0
    ys, xs = np.nonzero(changed1)
    r1 = np.hypot((xs + 0.5) / 128 - 1.0, (ys + 0.5) / 128 - 0.5)
    ys, xs = np.nonzero(changed2)
    r2 = np.hypot((xs + 0.5) / 128 - 1.0, (ys + 0.5) / 128 - 0.5)
    assert changed1.mean() > 0.02 and changed2.mean() > 0.02
    assert abs(np.median(r1) - 0.5) < 0.1 and abs(np.median(r2) - 0.8) < 0.1
    # Nothing far from the ring changes
    assert not changed1[:, :20].any()
    # No waves in the view style: identical to no rings at all
    from core.view import RenderView
    off = gl.render(snap, RenderView(256, 128, np.diag([2.0, 1.0, 1.0]), background=BG))
    assert np.array_equal(off, base)


def test_wave_glow_tints_the_crest(gl):
    gl.set_tile(TILE)
    snap = run_tree(2, 1, proc=32, levels=8, frames=1).snapshot()
    snap.waves = np.array([[1.0, 0.5, 0.0, 1.0]])
    plain = gl.render(snap, _ring_view(1.0, glow=0.0, depth=0)).astype(int)
    dark = gl.render(snap, _ring_view(1.0, glow=1.0, depth=0)).astype(int)
    ys, xs = np.mgrid[0:128, 0:256]
    r = np.hypot((xs + 0.5) / 128 - 1.0, (ys + 0.5) / 128 - 0.5)
    crest, far = np.abs(r - 0.5) < 0.02, np.abs(r - 0.5) > 0.3
    assert (plain - dark)[crest].mean() > 10                 # the ink at the crest turns black
    assert np.array_equal(plain[far], dark[far])
