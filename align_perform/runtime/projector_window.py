"""One projector's output: an OpenGL window, full screen on its display or a test window.

Draws (in order) the ornament or the test pattern's blended grey field, the
pattern lines, the identify number, the HUD, and the blackout fade.  All state
comes from the ProjectorOutputs manager; the window only renders it.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtGui import QSurfaceFormat

from align_perform.runtime import test_patterns
from align_perform.runtime.qt_gl import RendererWindow
from core.view import projector_view

BLACK = np.array([[[0, 0, 0, 255]]], np.uint8)
TEST_GREY = (150, 150, 150)


class ProjectorWindow(RendererWindow):
    def __init__(self, manager, proj_id: str, vsync: bool = True) -> None:
        super().__init__()
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setSwapInterval(1 if vsync else 0)
        self.setFormat(fmt)
        self.manager = manager
        self.proj_id = proj_id
        self.state = "hidden"
        self.placed_output = None          # OutputSpec this window was placed for
        self._tile_id = None
        self._pattern_key = None
        self._identify_key = None
        self._hud_key = None
        self._black_ready = False
        self.last_draw = 0.0

    def on_gl_ready(self) -> None:
        self._tile_id = None
        self._pattern_key = self._identify_key = self._hud_key = None
        self._black_ready = False

    def draw(self, fbo, w: int, h: int) -> None:
        m = self.manager
        r = self.renderer
        proj = m.projector(self.proj_id)
        self.last_draw = time.perf_counter()
        if proj is None:
            fbo.clear(0, 0, 0, 1)
            return
        tile = m.tile
        if tile is not None and id(tile) != self._tile_id:
            r.set_tile(tile)
            self._tile_id = id(tile)
        number = m.number(self.proj_id)

        if m.test_pattern:
            view = projector_view(m.scene, proj, w, h, params=m.params)
            view.flat = TEST_GREY
            r.render(fbo, view)
            key = (m.geometry_version, w, h, m.corner_sel.get(self.proj_id), number)
            rgba = None
            if key != self._pattern_key:
                rgba = test_patterns.test_pattern(m.scene, proj, w, h, number, m.corner_sel.get(self.proj_id))
                self._pattern_key = key
            r.draw_image(fbo, "pattern", rgba)
        else:
            if m.snapshot is not None:
                r.upload(m.snapshot)
            r.render(fbo, projector_view(m.scene, proj, w, h, params=m.params))

        if m.identify_until > time.perf_counter():
            key = (number, proj.name, w, h)
            rgba = None
            if key != self._identify_key:
                rgba = test_patterns.identify(number, proj.name or proj.id, w, h)
                self._identify_key = key
            r.draw_image(fbo, "identify", rgba)

        if m.hud_lines:
            key = (tuple(m.hud_lines), w, h)
            rgba = None
            if key != self._hud_key:
                rgba = test_patterns.hud(m.hud_lines, w, h)
                self._hud_key = key
            r.draw_image(fbo, "hud", rgba)

        if m.level < 1.0:
            r.draw_image(fbo, "black", None if self._black_ready else BLACK, opacity=1.0 - m.level)
            self._black_ready = True

    def keyPressEvent(self, event) -> None:
        self.manager.handle_key(self.proj_id, event)

    def closeEvent(self, event) -> None:
        if self.manager.closing:
            event.accept()
            return
        event.ignore()
        self.manager.window_closed(self.proj_id)
