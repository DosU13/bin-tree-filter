"""Looks: named parameter sets saved in the scene, applied instantly or crossfaded.

perform.py plays the same looks with its 1-9 keys, in this list's order.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QInputDialog, QListWidget, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from align_perform.panels.widgets import HintLabel, dspin
from core.looks import LookFader


class LooksPanel(QWidget):
    def __init__(self, doc, window, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.window = window
        self.list = QListWidget()
        self.fade = dspin(0.0, 60.0, 0.5, 1, " s")
        self.fade.setValue(2.0)
        b_save = QPushButton("Save current as…")
        b_update = QPushButton("Update")
        b_apply = QPushButton("Apply now")
        b_cross = QPushButton("Crossfade")
        b_rename = QPushButton("Rename…")
        b_delete = QPushButton("Delete")
        b_update.setToolTip("Overwrite the selected look with the current settings")
        row1 = QHBoxLayout()
        row1.addWidget(b_save)
        row1.addWidget(b_update)
        row2 = QHBoxLayout()
        row2.addWidget(b_apply)
        row2.addWidget(b_cross)
        row3 = QHBoxLayout()
        row3.addWidget(b_rename)
        row3.addWidget(b_delete)
        form = QFormLayout()
        form.addRow("Crossfade time", self.fade)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(form)
        lay.addLayout(row3)
        lay.addWidget(HintLabel("In perform.py the keys 1–9 crossfade to the first nine looks."))

        b_save.clicked.connect(self.save_as)
        b_update.clicked.connect(self.update_selected)
        b_apply.clicked.connect(lambda: self._with_selected(self.doc.apply_look))
        b_cross.clicked.connect(lambda: self._with_selected(lambda n: self.crossfade(n, self.fade.value())))
        b_rename.clicked.connect(self.rename)
        b_delete.clicked.connect(lambda: self._with_selected(self.doc.delete_look))
        self.list.itemDoubleClicked.connect(lambda item: self.crossfade(item.text(), self.fade.value()))
        doc.looksChanged.connect(self.refresh)

        self._fader: LookFader | None = None
        self._fade_target = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._step)
        self.refresh()

    def refresh(self) -> None:
        current = self.list.currentItem().text() if self.list.currentItem() else None
        self.list.clear()
        for i, name in enumerate(self.doc.scene.looks):
            self.list.addItem(name)
            if name == current:
                self.list.setCurrentRow(i)

    def _with_selected(self, fn) -> None:
        item = self.list.currentItem()
        if item is not None:
            fn(item.text())

    def save_as(self) -> None:
        name, ok = QInputDialog.getText(self, "Save look", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.doc.scene.looks and QMessageBox.question(
                self, "Save look", f"Replace the look “{name}”?") != QMessageBox.StandardButton.Yes:
            return
        self.doc.save_look(name)

    def update_selected(self) -> None:
        self._with_selected(self.doc.save_look)

    def rename(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name, ok = QInputDialog.getText(self, "Rename look", "New name:", text=item.text())
        if ok:
            self.doc.rename_look(item.text(), name.strip())

    # -- crossfade -----------------------------------------------------------

    def crossfade(self, name: str, seconds: float) -> None:
        """Glide the live output to look *name*; the document changes once, at the end."""
        target = self.doc.scene.looks.get(name)
        if target is None:
            return
        if seconds <= 0:
            self.doc.apply_look(name)
            return
        self._fader = LookFader(self.doc.scene.params, target, seconds, time.perf_counter())
        self._fade_target = name
        self._timer.start()

    @property
    def fading(self) -> bool:
        return self._fader is not None

    def _step(self) -> None:
        if self._fader is None:
            self._timer.stop()
            return
        now = time.perf_counter()
        values = self._fader.values(now)
        self.window.session.preview_params(values)
        if self.window.outputs is not None:
            self.window.outputs.preview_params(values)
        if self._fader.done(now):
            self._fader = None
            self._timer.stop()
            self.doc.apply_look(self._fade_target)
