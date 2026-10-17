"""LiveSession: keeps a PipelineRunner in step with the document and renders canvas previews.

Processing runs on the pipeline thread; this object lives on the Qt thread and
polls for new results ~30 times a second.  The ornament preview is rendered on
the GPU (offscreen) for exactly the world rectangle the canvas shows, at the
canvas's pixel size, so it stays sharp at any zoom.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QRectF, QTimer, Signal
from PySide6.QtGui import QImage

from core.engine import load_tile
from core.pipeline import FrameResult, PipelineRunner
from core.scene import Scene
from core.view import projector_view, stage_view, world_view

log = logging.getLogger(__name__)

PREVIEW_MODES = ("ornament", "input", "labels", "cameras")


def bgr_to_qimage(img: np.ndarray) -> QImage:
    """BGR (or gray) uint8 numpy image -> owned QImage."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8).copy()
    return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888).copy()


class LiveSession(QObject):
    frameReady = Signal(object)      # FrameResult, once per new pipeline frame
    previewReady = Signal()          # preview_image / preview_rect updated
    message = Signal(str)            # status-bar text (errors, tile problems)

    def __init__(self, scene: Scene, parent: QObject | None = None, max_fps: float = 60.0) -> None:
        super().__init__(parent)
        self.scene = scene.copy()
        self.runner = PipelineRunner(self.scene, max_fps=max_fps).start()
        self.last: FrameResult | None = None
        self.preview_mode = "ornament"
        self.preview_image: QImage | None = None
        self.preview_rect = QRectF()
        self._view_rect = QRectF()
        self._view_size = (0, 0)
        self._tile_key = None
        self.tile: np.ndarray | None = None
        self.gl = None
        try:
            from align_perform.runtime.gl_renderer import OffscreenGL
            self.gl = OffscreenGL()
        except Exception as exc:
            log.warning("no GPU preview: %s", exc)
            self.message.emit(f"No GPU preview ({exc})")
        self._load_tile()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- document sync -------------------------------------------------------

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene.copy()
        self.runner.set_scene(self.scene)
        self._load_tile()

    @property
    def params(self) -> dict:
        return self.scene.params

    def _load_tile(self) -> None:
        key = (self.scene.resolve(self.params["tile"]), self.params["tile_crop"])
        if key == self._tile_key:
            return
        try:
            tile = load_tile(key[0], crop=key[1])
        except Exception as exc:
            self.message.emit(f"Ornament tile: {exc}")
            if self.tile is not None:
                return
            tile = np.zeros((64, 64), np.float32)
        self._tile_key = key
        self.tile = tile
        if self.gl is not None:
            self.gl.set_tile(tile)
        self.tileChanged()

    def tileChanged(self) -> None:
        """Hook for projector windows (they upload their own copy)."""

    # -- preview -------------------------------------------------------------

    def set_preview_mode(self, mode: str) -> None:
        self.preview_mode = mode
        if self.last is not None:
            self._render_preview(self.last)

    def set_view(self, world_rect: QRectF, size: tuple[int, int]) -> None:
        """Canvas viewport changed: visible world rectangle and its size in pixels."""
        self._view_rect = QRectF(world_rect)
        self._view_size = size
        if self.last is not None:
            self._render_preview(self.last)

    def _render_preview(self, r: FrameResult) -> None:
        st = self.scene.stage
        stage_rect = QRectF(st.x, st.y, st.width, st.height)
        if self.preview_mode == "ornament":
            vr, (vw, vh) = self._view_rect, self._view_size
            if self.gl is None or vw < 2 or vh < 2 or vr.isEmpty():
                return
            # Never more pixels than the screen shows, never fewer than useful
            scale = min(1.0, 2400 / max(vw, vh))
            w, h = max(2, int(vw * scale)), max(2, int(vh * scale))
            view = world_view(self.scene, vr.x(), vr.y(), vr.width(), vr.height(), w, h)
            view.outside = (40, 40, 40)
            img = self.gl.render(r.snapshot, view)
            self.preview_image = bgr_to_qimage(img)
            self.preview_rect = QRectF(vr)
        elif self.preview_mode == "input":
            self.preview_image = bgr_to_qimage(r.stage_input)
            self.preview_rect = stage_rect
        elif self.preview_mode == "labels":
            self.preview_image = bgr_to_qimage(r.labels_image)
            self.preview_rect = stage_rect
        else:
            self.preview_image = None
        self.previewReady.emit()

    def _tick(self) -> None:
        r = self.runner.latest()
        if r is None or r is self.last:
            return
        self.last = r
        self._render_preview(r)
        self.frameReady.emit(r)

    # -- misc ----------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self.runner.paused

    def set_paused(self, paused: bool) -> None:
        self.runner.paused = paused

    def render_projector(self, projector, width: int | None = None, height: int | None = None) -> np.ndarray | None:
        """Current frame as one projector would show it (BGR), for snapshots."""
        if self.gl is None or self.last is None:
            return None
        return self.gl.render(self.last.snapshot, projector_view(self.scene, projector, width, height))

    def render_stage(self, width: int) -> np.ndarray | None:
        if self.gl is None or self.last is None:
            return None
        return self.gl.render(self.last.snapshot, stage_view(self.scene, width))

    def close(self) -> None:
        self._timer.stop()
        self.runner.stop()
        if self.gl is not None:
            self.gl.release()
            self.gl = None
