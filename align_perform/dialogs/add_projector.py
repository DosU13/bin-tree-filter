"""Add a projector: a connected display (full screen) or a test window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout)

from align_perform.panels.widgets import HintLabel, ispin
from align_perform.runtime.screens import list_screens
from core.scene import OutputSpec


class AddProjectorDialog(QDialog):
    def __init__(self, parent=None, name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add projector")
        self.result_output: OutputSpec | None = None
        self.list = QListWidget()
        refresh = QPushButton("Look for displays again")
        refresh.clicked.connect(self._fill)
        self.w, self.h = ispin(16, 16384), ispin(16, 16384)
        self.w.setValue(1920)
        self.h.setValue(1080)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Left")
        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Window width", self.w)
        form.addRow("Window height", self.h)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Where does this projector's image go?"))
        lay.addWidget(self.list, 1)
        lay.addWidget(refresh)
        lay.addLayout(form)
        lay.addWidget(HintLabel("A display shows the image full screen. A window is for testing on one "
                                "monitor; its size sets the projector resolution."))
        lay.addWidget(buttons)
        self.list.currentItemChanged.connect(self._changed)
        self._fill()
        self.resize(480, 420)

    def _fill(self) -> None:
        self.list.clear()
        item = QListWidgetItem("Window (testing)")
        item.setData(Qt.ItemDataRole.UserRole, None)
        self.list.addItem(item)
        for info in list_screens():
            item = QListWidgetItem("Display " + info.label())
            item.setData(Qt.ItemDataRole.UserRole, info)
            self.list.addItem(item)
        # Prefer a display that isn't the operator's main screen
        best = next((i for i in range(1, self.list.count())
                     if not self.list.item(i).data(Qt.ItemDataRole.UserRole).primary), 0)
        self.list.setCurrentRow(best)

    def _changed(self, cur, prev=None) -> None:
        info = cur.data(Qt.ItemDataRole.UserRole) if cur else None
        self.w.setEnabled(info is None)
        self.h.setEnabled(info is None)
        if info is not None:
            self.w.setValue(info.width)
            self.h.setValue(info.height)

    def _accept(self) -> None:
        cur = self.list.currentItem()
        info = cur.data(Qt.ItemDataRole.UserRole) if cur else None
        if info is None:
            self.result_output = OutputSpec(kind="window", width=self.w.value(), height=self.h.value())
        else:
            self.result_output = OutputSpec(kind="screen", screen_name=info.name, screen_serial=info.serial,
                                            screen_index=info.index, width=info.width, height=info.height)
        self.accept()
