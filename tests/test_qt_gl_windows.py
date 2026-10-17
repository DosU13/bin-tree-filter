"""On-screen Qt OpenGL windows hosting the GPU renderer (opens real windows briefly)."""

import os
import time

import numpy as np
import pytest

from core.engine import load_tile
from tests.engine_cases import ROOT
from tests.test_engine_multicell import run_tree

pytest.importorskip("PySide6")

pytestmark = [pytest.mark.gui, pytest.mark.gpu]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    from align_perform.runtime.qt_gl import configure_opengl
    app = QApplication.instance()
    if app is None:
        configure_opengl(vsync=False)
        app = QApplication([])
    return app


def test_two_windows_render_like_offscreen(qapp):
    from align_perform.runtime.gl_renderer import OffscreenGL, RenderView
    from align_perform.runtime.qt_gl import RendererWindow, qimage_to_bgr

    tile = load_tile(os.path.join(ROOT, "ornam.jpg"))
    tree = run_tree(2, 1, proc=32, levels=8, frames=4)
    snap = tree.snapshot()
    jitter = np.array([[1.0, 0, 0.000123], [0, 1.0, 0.000217], [0, 0, 1]])

    class Win(RendererWindow):
        def __init__(self, cell):
            super().__init__()
            self.cell = cell

        def on_gl_ready(self):
            self.renderer.set_tile(tile)

        def view(self, w, h):
            m = jitter @ np.array([[1.0, 0, self.cell], [0, 1.0, 0], [0, 0, 1]])
            return RenderView(w, h, m, mip=False)

        def draw(self, fbo, w, h):
            self.renderer.upload(snap)
            self.renderer.render(fbo, self.view(w, h))

    wins = [Win(0), Win(1)]
    for i, w in enumerate(wins):
        w.resize(160, 160)
        w.setPosition(100 + 200 * i, 100)
        w.show()
    deadline = time.time() + 5
    while time.time() < deadline and min(w.frames for w in wins) < 3:
        qapp.processEvents()
        for w in wins:
            w.update()
        time.sleep(0.01)
    assert min(w.frames for w in wins) >= 3

    off = OffscreenGL()
    off.set_tile(tile)
    for w in wins:
        img = qimage_to_bgr(w.grabFramebuffer())
        pw, ph = w.pixel_size()
        assert img.shape[:2] == (ph, pw)
        ref = off.render(snap, w.view(pw, ph))
        assert np.abs(img.astype(int) - ref.astype(int)).max() <= 1
    off.release()
    for w in wins:
        w.release_gl()
        w.close()
