"""GPU renderer timings at projector resolutions (called from tools.bench --gpu)."""

from __future__ import annotations

import time

import numpy as np

from core.engine import OrnamentTree


def _timed_frames(gl, fbo, view, frames: int) -> float:
    r = gl.renderer
    with gl.ctx:
        fbo.use()
        r.render(fbo, view)
        gl.ctx.finish()
        t = time.perf_counter()
        for _ in range(frames):
            r.render(fbo, view)
        gl.ctx.finish()
    return (time.perf_counter() - t) / frames * 1000


def bench_gpu(tile: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> None:
    from align_perform.runtime.gl_renderer import OffscreenGL, RenderView

    gl = OffscreenGL()
    gl.set_tile(tile)
    print("GPU:", gl.ctx.info["GL_RENDERER"])

    # 3 cells side by side, deep tree: worst case for the per-pixel walk
    tree = OrnamentTree(3, 1, 12, 128)
    tree.max_depth = 12
    tree.threshold = 0.95
    wide = np.concatenate([labels] * 3, axis=1)
    for _ in range(3):
        tree.update(wide, palette, 1 / 30)
    snap = tree.snapshot()
    with gl.ctx:
        gl.renderer.upload(snap)
    print(f"stage 3x1 cells, {tree.leaf_count} leaves")

    for w, h, label in ((1920, 1080, "1080p"), (3840, 2160, "4K")):
        with gl.ctx:
            fbo = gl.ctx.simple_framebuffer((w, h), components=4)
        # Each projector shows 1.5 cells of the 3-cell stage
        m = np.diag([1.5, 1.0, 1.0])
        for opts in (dict(), dict(cord=True), dict(supersample=1)):
            view = RenderView(w, h, m, **opts)
            ms = _timed_frames(gl, fbo, view, 30)
            name = ", ".join(f"{k}={v}" for k, v in opts.items()) or "default (ss=2, mip)"
            print(f"gpu {label} {name}: {ms:.2f} ms/frame ({1000 / ms:.0f} fps per output)")
        with gl.ctx:
            fbo.release()

    t = time.perf_counter()
    for _ in range(20):
        snap.version += 1
        with gl.ctx:
            gl.renderer.upload(snap)
    print(f"upload {snap.n_nodes} nodes: {(time.perf_counter() - t) / 20 * 1000:.2f} ms")
    gl.release()
