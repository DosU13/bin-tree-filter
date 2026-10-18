"""Small widget helpers shared by the panels."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QWidget)


def dspin(lo: float, hi: float, step: float = 1.0, decimals: int = 1, suffix: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setKeyboardTracking(False)       # commit on Enter / focus-out, not every keystroke
    w.setAccelerated(True)
    if suffix:
        w.setSuffix(suffix)
    return w


def ispin(lo: int, hi: int, step: int = 1, suffix: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setKeyboardTracking(False)
    w.setAccelerated(True)
    if suffix:
        w.setSuffix(suffix)
    return w


def combo(items, data=None) -> QComboBox:
    w = QComboBox()
    for i, text in enumerate(items):
        w.addItem(str(text), data[i] if data is not None else text)
    return w


def set_combo_data(w: QComboBox, value) -> None:
    i = w.findData(value)
    if i >= 0:
        w.setCurrentIndex(i)


def set_quiet(widget, value) -> None:
    """Set a widget's value without emitting change signals."""
    widget.blockSignals(True)
    try:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            widget.setValue(value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            set_combo_data(widget, value)
        elif isinstance(widget, ColorButton):
            widget.set_color(value)
        elif hasattr(widget, "setText"):
            widget.setText(str(value))
    finally:
        widget.blockSignals(False)


class ColorButton(QPushButton):
    """Swatch button holding a '#rrggbb' colour; click opens a picker."""

    colorChanged = Signal(str)

    def __init__(self, value: str = "#000000", parent=None) -> None:
        super().__init__(parent)
        self._value = value
        self.setFixedHeight(24)
        self.clicked.connect(self._pick)
        self.set_color(value)

    def value(self) -> str:
        return self._value

    def set_color(self, value: str) -> None:
        self._value = value
        c = QColor(value)
        text = "#000" if c.lightness() > 140 else "#fff"
        self.setStyleSheet(f"background:{value}; color:{text}; border:1px solid #555;")
        self.setText(value)

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self._value), self, "Pick colour")
        if c.isValid():
            self.set_color(c.name())
            self.colorChanged.emit(c.name())


class SliderSpin(QWidget):
    """Slider + spin box for one number; emits valueChanged(float) and editingDone()."""

    valueChanged = Signal(float)

    def __init__(self, lo: float, hi: float, step: float, decimals: int, is_int: bool, parent=None) -> None:
        super().__init__(parent)
        self.lo, self.step, self.is_int = lo, step, is_int
        self.slider = QSlider(Qt.Orientation.Horizontal)
        n = max(1, round((hi - lo) / step))
        self.slider.setRange(0, n)
        self.spin = ispin(int(lo), int(hi), int(step)) if is_int else dspin(lo, hi, step, decimals)
        self.spin.setFixedWidth(78)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(round((v - self.lo) / self.step))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def _from_slider(self, i: int) -> None:
        v = self.lo + i * self.step
        v = int(round(v)) if self.is_int else round(v, 6)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _from_spin(self, v) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def status_dot(color: str, text: str) -> str:
    return f'<span style="color:{color}">●</span> {text}'


class HintLabel(QLabel):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet("color:#999;")
