"""Scene list: stage, cameras and projectors, with enable checkboxes and a context menu."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

ROLE = Qt.ItemDataRole.UserRole


class SceneTree(QTreeWidget):
    addCamera = Signal()
    addProjector = Signal()
    focusItem = Signal(str, str)        # kind, id: fit the canvas to it

    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setHeaderHidden(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self._syncing = False
        self.stage = QTreeWidgetItem(["Stage"])
        self.stage.setData(0, ROLE, ("stage", ""))
        self.cameras = QTreeWidgetItem(["Cameras"])
        self.projectors = QTreeWidgetItem(["Projectors"])
        for top in (self.cameras, self.projectors):
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.addTopLevelItems([self.stage, self.cameras, self.projectors])
        self.itemSelectionChanged.connect(self._on_select)
        self.itemChanged.connect(self._on_check)
        self.itemDoubleClicked.connect(self._on_double)
        doc.itemsChanged.connect(self.rebuild)
        doc.itemChanged.connect(lambda *_: self.rebuild())
        doc.selectionChanged.connect(self._sync_selection)
        self.rebuild()

    def rebuild(self) -> None:
        self._syncing = True
        try:
            scene = self.doc.scene
            for top, items, kind in ((self.cameras, scene.cameras, "camera"),
                                     (self.projectors, scene.projectors, "projector")):
                top.takeChildren()
                for spec in items:
                    label = spec.name or spec.id
                    extra = spec.source.label() if kind == "camera" else (
                        f"{spec.output.width}×{spec.output.height}")
                    child = QTreeWidgetItem([f"{label}  —  {extra}"])
                    child.setData(0, ROLE, (kind, spec.id))
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked if spec.enabled else Qt.CheckState.Unchecked)
                    top.addChild(child)
                top.setText(0, f"{'Cameras' if kind == 'camera' else 'Projectors'} ({len(items)})")
                top.setExpanded(True)
        finally:
            self._syncing = False
        self._sync_selection()

    def _find(self, key) -> QTreeWidgetItem | None:
        if key is None:
            return None
        if key[0] == "stage":
            return self.stage
        top = self.cameras if key[0] == "camera" else self.projectors
        for i in range(top.childCount()):
            if top.child(i).data(0, ROLE) == key:
                return top.child(i)
        return None

    def _sync_selection(self) -> None:
        self._syncing = True
        try:
            self.clearSelection()
            item = self._find(self.doc.selection)
            if item is not None:
                item.setSelected(True)
                self.scrollToItem(item)
        finally:
            self._syncing = False

    def _on_select(self) -> None:
        if self._syncing:
            return
        items = self.selectedItems()
        key = items[0].data(0, ROLE) if items else None
        if key:
            self.doc.select(*key)
        else:
            self.doc.select(None)

    def _on_check(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing:
            return
        key = item.data(0, ROLE)
        if not key or key[0] == "stage":
            return
        on = item.checkState(0) == Qt.CheckState.Checked
        self.doc.update_item(key[0], key[1], lambda s: setattr(s, "enabled", on), "Enable" if on else "Disable")

    def _on_double(self, item: QTreeWidgetItem, column: int) -> None:
        key = item.data(0, ROLE)
        if key:
            self.focusItem.emit(*key)

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        key = item.data(0, ROLE) if item else None
        menu = QMenu(self)
        menu.addAction("Add camera…", self.addCamera.emit)
        menu.addAction("Add projector…", self.addProjector.emit)
        if key and key[0] in ("camera", "projector"):
            menu.addSeparator()
            menu.addAction("Duplicate", lambda: self.doc.duplicate_item(*key))
            menu.addAction("Delete", lambda: self.doc.remove_item(*key))
        menu.exec(self.viewport().mapToGlobal(pos))
