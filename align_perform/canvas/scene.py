"""CanvasScene: keeps canvas items in step with the SceneDocument and routes edits back.

Clicking picks the topmost item, except that the selected item keeps the click
while the pointer is over it (so a camera under a projector can be dragged once
selected), and Alt+click cycles through everything under the pointer.
"""

from __future__ import annotations

import itertools

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsScene

from align_perform.canvas.items import CameraItem, PlacedItem, ProjectorItem, STATUS_COLORS, StageItem, quad_polygon
from align_perform.canvas.snapping import SNAP_PX, Snapper, targets_for

WORLD_MARGIN = 4000


class CanvasScene(QGraphicsScene):
    viewRectNeeded = Signal()

    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.stage_item = StageItem(self)
        self.addItem(self.stage_item)
        self.items_by_key: dict[tuple[str, str], PlacedItem] = {}
        self.show_camera_images = False
        self.camera_opacity = 0.75
        self.show_grid = True
        self.show_cameras = True
        self.show_projectors = True
        self.snap = True
        self._guides: tuple[list, list] = ([], [])
        self._drag_ids = itertools.count()
        self._drag_key = None
        self._syncing = False
        self._statuses: dict[str, str] = {}

        doc.itemsChanged.connect(self.sync_items)
        doc.itemChanged.connect(self._on_item_changed)
        doc.stageChanged.connect(self._on_stage_changed)
        doc.selectionChanged.connect(self._on_doc_selection)
        self.selectionChanged.connect(self._on_scene_selection)
        self._on_stage_changed()
        self.sync_items()

    # -- view info -----------------------------------------------------------

    def view_scale(self) -> float:
        views = self.views()
        return views[0].transform().m11() if views else 1.0

    # -- document -> items ---------------------------------------------------

    def _on_stage_changed(self) -> None:
        st = self.doc.scene.stage
        self.stage_item.set_stage(st)
        self.setSceneRect(QRectF(st.x - WORLD_MARGIN, st.y - WORLD_MARGIN,
                                 st.width + 2 * WORLD_MARGIN, st.height + 2 * WORLD_MARGIN))
        self.update()

    def sync_items(self) -> None:
        scene = self.doc.scene
        wanted = {("camera", c.id): c for c in scene.cameras}
        wanted.update({("projector", p.id): p for p in scene.projectors})
        for key in list(self.items_by_key):
            if key not in wanted:
                self.removeItem(self.items_by_key.pop(key))
        for key, spec in wanted.items():
            item = self.items_by_key.get(key)
            if item is None:
                item = CameraItem(self, spec.id) if key[0] == "camera" else ProjectorItem(self, spec.id)
                self.items_by_key[key] = item
                self.addItem(item)
            self._update_item(item, spec)
        self._on_doc_selection()
        self.update()

    def _on_item_changed(self, kind: str, item_id: str) -> None:
        item = self.items_by_key.get((kind, item_id))
        if item is not None:
            self._update_item(item, self.doc.scene.item(kind, item_id))
            self.update()

    def _update_item(self, item: PlacedItem, spec) -> None:
        if item._drag is None:                   # don't fight the mouse
            item.set_transform(spec.transform)
        if isinstance(item, CameraItem):
            from core.geometry import display_size
            w, h = display_size(*spec.frame_size(), spec.transform.rotation)
            item.aspect = w / h
            res = f"{spec.source.width}×{spec.source.height}" if spec.source.width else "?"
            status = "disabled" if not spec.enabled else self._statuses.get(spec.id, "starting")
            item.label.set_text(f"{spec.name or spec.id}  {res}", STATUS_COLORS.get(status))
            item.setVisible(self.show_cameras)
        else:
            item.aspect = spec.output.width / spec.output.height
            out = spec.output
            where = {"screen": out.screen_name or f"screen {out.screen_index}", "window": "window",
                     "none": "off"}[out.kind]
            item.label.set_text(f"{spec.name or spec.id}  {out.width}×{out.height}  {where}",
                                STATUS_COLORS["ok" if spec.enabled else "disabled"])
            item.setVisible(self.show_projectors)
        item.setOpacity(1.0 if spec.enabled else 0.45)

    def set_camera_status(self, statuses: dict[str, str]) -> None:
        if statuses == self._statuses:
            return
        self._statuses = dict(statuses)
        for c in self.doc.scene.cameras:
            item = self.items_by_key.get(("camera", c.id))
            if item is not None:
                self._update_item(item, c)

    def set_thumbnails(self, thumbs: dict) -> None:
        if not self.show_camera_images:
            return
        for cid, img in thumbs.items():
            item = self.items_by_key.get(("camera", cid))
            if item is not None:
                item.set_frame(img)

    def set_visibility(self, cameras: bool | None = None, projectors: bool | None = None) -> None:
        if cameras is not None:
            self.show_cameras = cameras
        if projectors is not None:
            self.show_projectors = projectors
        for (kind, _), item in self.items_by_key.items():
            item.setVisible(self.show_cameras if kind == "camera" else self.show_projectors)
        self.update()

    # -- selection -----------------------------------------------------------

    def select_item(self, kind: str, item_id: str) -> None:
        self.doc.select(kind, item_id)

    def _on_doc_selection(self) -> None:
        self._syncing = True
        try:
            sel = self.doc.selection
            for key, item in self.items_by_key.items():
                item.setSelected(sel == key)
            self.stage_item.setSelected(sel is not None and sel[0] == "stage")
        finally:
            self._syncing = False

    def _on_scene_selection(self) -> None:
        if self._syncing:
            return
        chosen = [i for i in self.selectedItems() if isinstance(i, (PlacedItem, StageItem))]
        if not chosen:
            self.doc.select(None)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            under = [i for i in self.items(pos) if isinstance(i, PlacedItem) and i.isVisible()
                     and i.shape().contains(pos)]
            sel = self.doc.selection
            current = self.items_by_key.get(sel) if sel else None
            if event.modifiers() & Qt.KeyboardModifier.AltModifier and under:
                # Cycle to the next item under the pointer
                idx = under.index(current) + 1 if current in under else 0
                target = under[idx % len(under)]
                self.select_item(target.kind, target.item_id)
                event.accept()
                return
            if current is not None and (current in under or current.handle_at(pos)):
                # The selected item keeps the gesture even when another item is on top;
                # move / release are forwarded by this scene
                current.mousePressEvent(event)
                if event.isAccepted():
                    self._grabbed = current
                    return
            if not under and not self.stage_item.shape().contains(pos):
                self.doc.select(None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        grabbed = getattr(self, "_grabbed", None)
        if grabbed is not None:
            grabbed.mouseMoveEvent(event)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        grabbed = getattr(self, "_grabbed", None)
        if grabbed is not None:
            self._grabbed = None
            grabbed.mouseReleaseEvent(event)
            return
        super().mouseReleaseEvent(event)

    # -- edits from items ----------------------------------------------------

    def begin_drag(self, kind: str, item_id: str) -> None:
        self._drag_key = f"drag-{kind}-{item_id}-{next(self._drag_ids)}"

    def snapper_for(self, kind: str, item_id: str, enabled: bool = True) -> Snapper:
        xs, ys = targets_for(self.doc.scene, exclude=(kind, item_id), cell_lines=self.show_grid)
        return Snapper(xs, ys, SNAP_PX / max(self.view_scale(), 1e-6), enabled=enabled and self.snap)

    def item_edited(self, kind: str, item_id: str, transform, final: bool) -> None:
        verb = "Move" if self._drag_key else "Edit"
        def mutate(spec, t=transform):
            spec.transform = t
        try:
            self.doc.update_item(kind, item_id, mutate, f"{verb} {kind} {item_id}", merge_key=self._drag_key)
        except KeyError:
            return
        if final:
            self._drag_key = None

    def nudge_selected(self, dx: float, dy: float) -> None:
        sel = self.doc.selection
        if not sel or sel[0] not in ("camera", "projector"):
            return
        def mutate(spec):
            t = spec.transform
            t.x += dx
            t.y += dy
            if t.corners is not None:
                t.corners = [[x + dx, y + dy] for x, y in t.corners]
        self.doc.update_item(sel[0], sel[1], mutate, "Nudge", merge_key=f"nudge-{sel[1]}")

    def set_guides(self, xs: list, ys: list) -> None:
        self._guides = (xs, ys)
        self.update()

    # -- overlays ------------------------------------------------------------

    def drawForeground(self, p: QPainter, rect: QRectF) -> None:
        scene = self.doc.scene
        # Projector overlaps (blend zones) and camera overlaps (feather zones)
        for kind, color, show in (("projector", QColor(255, 160, 40, 90), self.show_projectors),
                                  ("camera", QColor(80, 170, 255, 70), self.show_cameras)):
            if not show:
                continue
            items = scene.projectors if kind == "projector" else scene.cameras
            polys = [quad_polygon(i.transform.quad()) for i in items if i.enabled]
            brush = QBrush(color, Qt.BrushStyle.BDiagPattern)
            p.setPen(Qt.PenStyle.NoPen)
            for a in range(len(polys)):
                for b in range(a + 1, len(polys)):
                    inter = polys[a].intersected(polys[b])
                    if not inter.isEmpty():
                        p.setBrush(brush)
                        p.drawPolygon(inter)
        xs, ys = self._guides
        if xs or ys:
            pen = QPen(QColor(255, 60, 200), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            for x in xs:
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            for y in ys:
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
