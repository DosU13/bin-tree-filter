"""CanvasView: zoom, pan, fit, and the live preview image as background.

Mouse: wheel zooms at the pointer; middle-drag or Space+drag pans.
Keys (canvas focused): arrows nudge the selection (Shift = 10 px), F fits the
stage, and the live.py parameter keys (d/D, h/H, c/C, ...) change the look.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGraphicsView

from core import params as P

ZOOM_STEP = 1.15
MIN_SCALE, MAX_SCALE = 0.005, 40.0


class CanvasView(QGraphicsView):
    viewChanged = Signal(QRectF, int, int)     # visible world rect, viewport w, h (device px)
    message = Signal(str)

    def __init__(self, canvas, session_getter, parent=None) -> None:
        super().__init__(canvas, parent)
        self.canvas = canvas
        self._session = session_getter
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QColor(28, 28, 30))
        self._pan_from: QPointF | None = None
        self._space = False
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(30)
        self._emit_timer.timeout.connect(self._emit_view)

    # -- geometry ------------------------------------------------------------

    def visible_world_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def _emit_view(self) -> None:
        dpr = self.devicePixelRatioF()
        vp = self.viewport().rect()
        self.viewChanged.emit(self.visible_world_rect(), int(vp.width() * dpr), int(vp.height() * dpr))

    def schedule_view_update(self) -> None:
        self._emit_timer.start()

    def fit(self, rect: QRectF | None = None) -> None:
        if rect is None:
            rect = self.canvas.stage_item.rect()
            for item in self.canvas.items_by_key.values():
                if item.isVisible():
                    rect = rect.united(item.boundingRect())
        if rect.isEmpty():
            return
        m = 0.06 * max(rect.width(), rect.height())
        self.fitInView(rect.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        self.schedule_view_update()

    def zoom(self, factor: float) -> None:
        s = self.transform().m11()
        factor = max(MIN_SCALE / s, min(MAX_SCALE / s, factor))
        self.scale(factor, factor)
        self.schedule_view_update()
        self.canvas.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.schedule_view_update()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.schedule_view_update()

    # -- background preview --------------------------------------------------

    def drawBackground(self, p: QPainter, rect: QRectF) -> None:
        super().drawBackground(p, rect)
        s = self._session()
        if s is None or s.preview_image is None or s.preview_rect.isEmpty():
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s.preview_mode != "labels")
        p.drawImage(s.preview_rect, s.preview_image)
        p.restore()

    # -- mouse ---------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120
        if steps:
            self.zoom(ZOOM_STEP ** steps)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
                self._space and event.button() == Qt.MouseButton.LeftButton):
            self._pan_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_from is not None:
            d = event.position() - self._pan_from
            self._pan_from = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # -- keyboard ------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        step = 10.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
        arrows = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                  Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        if key in arrows:
            self.canvas.nudge_selected(*arrows[key])
            return
        if key == Qt.Key.Key_F and not mods & ~Qt.KeyboardModifier.KeypadModifier:
            self.fit()
            return
        text = event.text()
        if text and not (mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
            doc = self.canvas.doc
            hot = P.hotkeys(doc.scene.params["mode"])
            if text in hot:
                param, direction = hot[text]
                doc.nudge_param(param, direction)
                value = doc.scene.params[param]
                self.message.emit(f"{P.BY_KEY[param].label}: {value}")
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.unsetCursor()
            return
        super().keyReleaseEvent(event)
