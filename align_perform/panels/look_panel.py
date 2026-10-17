"""Look panel: every live.py option as a widget, generated from core.params.

Rows that only apply to one mode (colour / depth) are shown only in that mode.
Each label lists the option's hotkeys, which work while the canvas has focus.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from align_perform.panels.widgets import ColorButton, SliderSpin, combo, set_quiet
from core import params as P

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)"


class TilePicker(QWidget):
    """Ornament tile path + browse button + thumbnail."""

    def __init__(self, on_change, resolve) -> None:
        super().__init__()
        self.on_change, self.resolve = on_change, resolve
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        self.thumb = QLabel()
        self.thumb.setFixedSize(48, 48)
        self.thumb.setStyleSheet("background:#fff;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.thumb)
        lay.addWidget(self.edit, 1)
        lay.addWidget(browse)

    def setText(self, path: str) -> None:
        self.edit.setText(os.path.basename(path))
        self.edit.setToolTip(self.resolve(path))
        pix = QPixmap(self.resolve(path))
        if not pix.isNull():
            self.thumb.setPixmap(pix.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ornament tile (black ink on white)",
                                              os.path.dirname(self.edit.toolTip()), IMAGE_FILTER)
        if path:
            self.on_change(path)


class LookPanel(QScrollArea):
    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        self.widgets: dict[str, QWidget] = {}
        self.rows: dict[str, tuple[QFormLayout, QWidget]] = {}
        self.groups: dict[str, QGroupBox] = {}

        top = QHBoxLayout()
        reset = QPushButton("Reset all to defaults")
        reset.clicked.connect(lambda: self.doc.set_params(P.defaults(), "Reset look"))
        top.addStretch(1)
        top.addWidget(reset)
        lay.addLayout(top)

        for group in P.GROUPS:
            box = QGroupBox(group)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.groups[group] = box
            for p in P.PARAMS:
                if p.group != group:
                    continue
                w = self._make(p)
                label = QLabel(p.label + (f"  <span style='color:#888'>({' / '.join(p.keys)})</span>" if p.keys else ""))
                label.setToolTip(p.help)
                w.setToolTip(p.help)
                form.addRow(label, w)
                self.widgets[p.key] = w
                self.rows[p.key] = (form, label)
            lay.addWidget(box)
        lay.addStretch(1)
        self.setWidget(body)
        doc.paramsChanged.connect(self.refresh)
        self.refresh()

    def _make(self, p: P.Param) -> QWidget:
        key = p.key
        if p.kind in ("int", "float"):
            w = SliderSpin(p.min, p.max, p.step, p.decimals, p.kind == "int")
            w.valueChanged.connect(lambda v, k=key: self.doc.set_param(k, v))
        elif p.kind == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda v, k=key: self.doc.set_param(k, v))
        elif p.kind == "choice":
            w = combo(list(p.choices))
            w.currentIndexChanged.connect(lambda _, k=key, w=w: self.doc.set_param(k, w.currentData()))
        elif p.kind == "color":
            w = ColorButton(p.default)
            w.colorChanged.connect(lambda v, k=key: self.doc.set_param(k, v))
        else:
            w = TilePicker(lambda path, k=key: self.doc.set_param(k, path), self.doc.scene.resolve)
        return w

    def refresh(self) -> None:
        values = self.doc.scene.params
        mode = values["mode"]
        for key, w in self.widgets.items():
            v = values[key]
            if isinstance(w, SliderSpin):
                w.setValue(v)
            elif isinstance(w, TilePicker):
                w.setText(v)
            else:
                set_quiet(w, v)
            p = P.BY_KEY[key]
            visible = p.mode in (None, mode)
            form, label = self.rows[key]
            label.setVisible(visible)
            w.setVisible(visible)
        for group, box in self.groups.items():
            box.setVisible(any(P.BY_KEY[k].mode in (None, mode) for k in self.widgets if P.BY_KEY[k].group == group))
