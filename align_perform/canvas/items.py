"""Canvas items: the stage, cameras and projectors, drawn in world coordinates.

Items keep their geometry in scene (= world) coordinates with pos() at 0, 0, so
rectangles and 4-corner quads are handled the same way.  Mouse handling lives
here: move, 8-handle resize (aspect locked unless Shift; Alt = from centre),
corner dragging in corner-pin mode, all with snapping (Ctrl = no snap).
Edits are reported to the canvas (CanvasScene.item_edited), which turns them
into undoable document changes.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPolygonF,
                           QTransform)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from align_perform.canvas.handles import CORNERS, CURSORS, HANDLE_PX, handle_points, resize_rect
from core.scene import Transform

CAMERA_COLOR = QColor(80, 170, 255)
PROJECTOR_COLOR = QColor(255, 160, 40)
STAGE_COLOR = QColor(235, 235, 235)
STATUS_COLORS = {
    "ok": QColor(60, 200, 90), "starting": QColor(230, 190, 40), "stalled": QColor(240, 140, 30),
    "reconnecting": QColor(230, 60, 50), "stopped": QColor(140, 140, 140), "disabled": QColor(110, 110, 110),
}


def quad_polygon(quad) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in quad])


class Label(QGraphicsItem):
    """Screen-size text tag with a dark background (ignores zoom).

    anchor "top": the tag hangs below its position; "bottom": it sits above it.
    """

    HEIGHT = 18

    def __init__(self, parent: QGraphicsItem, color: QColor, anchor: str = "top") -> None:
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.text = ""
        self.color = color
        self.anchor = anchor
        self.dot: QColor | None = None
        self.font = QFont()
        self.font.setPixelSize(12)
        self._rect = QRectF()

    def set_text(self, text: str, dot: QColor | None = None) -> None:
        if text == self.text and dot == self.dot and not self._rect.isEmpty():
            return
        self.prepareGeometryChange()
        self.text, self.dot = text, dot
        width = QFontMetrics(self.font).horizontalAdvance(text) + (13 if dot else 0) + 12
        y = 0 if self.anchor == "top" else -self.HEIGHT
        self._rect = QRectF(0, y, width, self.HEIGHT)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, p: QPainter, option, widget=None) -> None:
        r = self._rect
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(r, 3, 3)
        x = r.left() + 6
        if self.dot is not None:
            p.setBrush(self.dot)
            p.drawEllipse(QPointF(x + 4, r.center().y()), 4, 4)
            x += 13
        p.setFont(self.font)
        p.setPen(self.color)
        p.drawText(QRectF(x, r.top(), r.right() - x, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text)


class PlacedItem(QGraphicsObject):
    """A camera or projector placed on the canvas."""

    color = CAMERA_COLOR
    kind = ""
    label_anchor = "top"

    def __init__(self, canvas, item_id: str) -> None:
        super().__init__()
        self.canvas = canvas
        self.item_id = item_id
        self.t = Transform()
        self.aspect: float | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.label = Label(self, QColor(240, 240, 240), anchor=self.label_anchor)
        self._drag = None

    # -- geometry ------------------------------------------------------------

    def quad(self) -> np.ndarray:
        return self.t.quad()

    def set_transform(self, t: Transform) -> None:
        self.prepareGeometryChange()
        self.t = copy.deepcopy(t)
        q = self.quad()
        y = q[:, 1].min() if self.label_anchor == "top" else q[:, 1].max()
        self.label.setPos(float(q[:, 0].min()), float(y))
        self.update()

    def _scale(self) -> float:
        return self.canvas.view_scale()

    def boundingRect(self) -> QRectF:
        q = self.quad()
        m = (HANDLE_PX + 4) / max(self._scale(), 1e-6)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        return QRectF(x0 - m, y0 - m, x1 - x0 + 2 * m, y1 - y0 + 2 * m)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(quad_polygon(self.quad()))
        path.closeSubpath()
        if self.isSelected():
            s = HANDLE_PX / max(self._scale(), 1e-6)
            for x, y in self.handles().values():
                path.addRect(QRectF(x - s, y - s, 2 * s, 2 * s))
        return path

    def handles(self) -> dict[str, tuple[float, float]]:
        if self.t.corners is not None:
            return {name: (float(x), float(y)) for name, (x, y) in zip(CORNERS, self.t.corners)}
        return handle_points(self.t.x, self.t.y, self.t.width, self.t.height)

    def handle_at(self, pos: QPointF) -> str | None:
        if not self.isSelected():
            return None
        r = HANDLE_PX / max(self._scale(), 1e-6)
        for name, (x, y) in self.handles().items():
            if abs(pos.x() - x) <= r and abs(pos.y() - y) <= r:
                return name
        return None

    # -- painting ------------------------------------------------------------

    def paint_content(self, p: QPainter) -> None:
        """Subclasses draw inside the quad (images, fills)."""

    def paint(self, p: QPainter, option, widget=None) -> None:
        poly = quad_polygon(self.quad())
        self.paint_content(p)
        sel = self.isSelected()
        pen = QPen(self.color.lighter(130) if sel else self.color, 2.5 if sel else 1.5)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(poly)
        if sel:
            s = HANDLE_PX / 2 / max(self._scale(), 1e-6)
            hp = QPen(QColor(20, 20, 20), 1)
            hp.setCosmetic(True)
            p.setPen(hp)
            p.setBrush(QColor(255, 255, 255))
            for x, y in self.handles().values():
                p.drawRect(QRectF(x - s, y - s, 2 * s, 2 * s))

    # -- interaction ---------------------------------------------------------

    def hoverMoveEvent(self, event) -> None:
        h = self.handle_at(event.scenePos())
        self.setCursor(CURSORS.get(h, Qt.CursorShape.SizeAllCursor) if h else Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self.canvas.select_item(self.kind, self.item_id)
        handle = self.handle_at(event.scenePos())
        self._drag = (handle, copy.deepcopy(self.t), event.scenePos())
        self.canvas.begin_drag(self.kind, self.item_id)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        handle, start, press = self._drag
        pos = event.scenePos()
        mods = event.modifiers()
        snapper = self.canvas.snapper_for(self.kind, self.item_id, enabled=not (mods & Qt.KeyboardModifier.ControlModifier))
        t = copy.deepcopy(start)
        guides_x, guides_y = [], []
        if handle is None:                                 # move
            dx, dy = pos.x() - press.x(), pos.y() - press.y()
            x0, y0, w, h = start.bounds()
            nx, ny, guides_x, guides_y = snapper.snap_move(x0 + dx, y0 + dy, w, h)
            dx, dy = nx - x0, ny - y0
            t.x, t.y = start.x + dx, start.y + dy
            if start.corners is not None:
                t.corners = [[cx + dx, cy + dy] for cx, cy in start.corners]
        elif start.corners is not None:                    # corner pin
            px, py, guides_x, guides_y = snapper.snap_point(pos.x(), pos.y())
            i = CORNERS.index(handle)
            t.corners = [list(c) for c in start.corners]
            t.corners[i] = [px, py]
        else:                                              # resize
            sx = handle in ("tl", "tr", "bl", "br", "l", "r")
            sy = handle in ("tl", "tr", "bl", "br", "t", "b")
            px, py, guides_x, guides_y = snapper.snap_point(pos.x(), pos.y(), sx, sy)
            free = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            aspect = None if free else (self.aspect or start.width / start.height)
            centre = bool(mods & Qt.KeyboardModifier.AltModifier)
            t.x, t.y, t.width, t.height = resize_rect((start.x, start.y, start.width, start.height),
                                                       handle, (px, py), aspect, centre)
        self.set_transform(t)
        self.canvas.set_guides(guides_x, guides_y)
        self.canvas.item_edited(self.kind, self.item_id, t, final=False)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.canvas.set_guides([], [])
        self.canvas.item_edited(self.kind, self.item_id, copy.deepcopy(self.t), final=True)


class CameraItem(PlacedItem):
    color = CAMERA_COLOR
    kind = "camera"

    def __init__(self, canvas, item_id: str) -> None:
        super().__init__(canvas, item_id)
        self.setZValue(2)
        self.image: QImage | None = None

    def set_frame(self, frame: np.ndarray | None) -> None:
        """Latest small camera frame (BGR, source orientation)."""
        if frame is None:
            self.image = None
        else:
            img = cv2.flip(frame, 1) if self.t.mirror else frame
            rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            if self.t.rotation in rot:
                img = cv2.rotate(img, rot[self.t.rotation])
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            fmt = QImage.Format.Format_Grayscale8 if img.ndim == 2 else QImage.Format.Format_BGR888
            self.image = QImage(img.data, w, h, img.strides[0], fmt).copy()
        self.update()

    def paint_content(self, p: QPainter) -> None:
        if self.image is None or not self.canvas.show_camera_images:
            return
        src = QPolygonF([QPointF(0, 0), QPointF(self.image.width(), 0),
                         QPointF(self.image.width(), self.image.height()), QPointF(0, self.image.height())])
        xf = QTransform()
        if QTransform.quadToQuad(src, quad_polygon(self.quad()), xf):
            p.save()
            p.setOpacity(self.canvas.camera_opacity)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setTransform(xf, True)
            p.drawImage(0, 0, self.image)
            p.restore()


class ProjectorItem(PlacedItem):
    color = PROJECTOR_COLOR
    kind = "projector"
    label_anchor = "bottom"

    def __init__(self, canvas, item_id: str) -> None:
        super().__init__(canvas, item_id)
        self.setZValue(3)

    def paint_content(self, p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        fill = QColor(self.color)
        fill.setAlpha(28 if self.isSelected() else 14)
        p.setBrush(fill)
        p.drawPolygon(quad_polygon(self.quad()))


class StageItem(QGraphicsObject):
    """Stage outline and cell grid (the preview image is drawn by the view's background)."""

    kind = "stage"

    def __init__(self, canvas) -> None:
        super().__init__()
        self.canvas = canvas
        self.stage = None
        self.setZValue(1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.label = Label(self, STAGE_COLOR, anchor="bottom")

    def set_stage(self, stage) -> None:
        self.prepareGeometryChange()
        self.stage = copy.deepcopy(stage)
        self.label.setPos(stage.x, stage.y)
        self.label.set_text(f"Stage {stage.cols}×{stage.rows} cells of {stage.cell_size:g} px")
        self.update()

    def rect(self) -> QRectF:
        s = self.stage
        return QRectF(s.x, s.y, s.width, s.height) if s else QRectF()

    def boundingRect(self) -> QRectF:
        return self.rect().adjusted(-2, -2, 2, 2)

    def shape(self) -> QPainterPath:
        # Only the outline band is clickable, so items on the stage stay easy to hit
        path = QPainterPath()
        r = self.rect()
        m = 6 / max(self.canvas.view_scale(), 1e-6)
        path.addRect(r.adjusted(-m, -m, m, m))
        inner = QPainterPath()
        inner.addRect(r.adjusted(m, m, -m, -m))
        return path.subtracted(inner)

    def mousePressEvent(self, event) -> None:
        self.canvas.select_item("stage", "")
        event.accept()

    def paint(self, p: QPainter, option, widget=None) -> None:
        if self.stage is None:
            return
        s = self.stage
        r = self.rect()
        if self.canvas.show_grid:
            pen = QPen(QColor(255, 255, 255, 70), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            for i in range(1, s.cols):
                x = s.x + i * s.cell_size
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            for j in range(1, s.rows):
                y = s.y + j * s.cell_size
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        pen = QPen(STAGE_COLOR if not self.isSelected() else QColor(255, 255, 140), 2 if self.isSelected() else 1.2)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
