"""Displays attached to this machine, and matching a saved projector output to one."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QGuiApplication, QScreen


@dataclass
class ScreenInfo:
    index: int
    name: str
    manufacturer: str
    model: str
    serial: str
    x: int
    y: int
    width: int          # physical pixels
    height: int
    dpr: float
    primary: bool

    def label(self) -> str:
        who = " ".join(s for s in (self.manufacturer, self.model) if s) or self.name
        tag = " (primary)" if self.primary else ""
        return f"{self.index}: {who} — {self.width}×{self.height}{tag}"


def list_screens() -> list[ScreenInfo]:
    primary = QGuiApplication.primaryScreen()
    out = []
    for i, s in enumerate(QGuiApplication.screens()):
        g = s.geometry()
        dpr = s.devicePixelRatio()
        out.append(ScreenInfo(i, s.name(), s.manufacturer(), s.model(), s.serialNumber(), g.x(), g.y(),
                              round(g.width() * dpr), round(g.height() * dpr), dpr, s is primary))
    return out


def find_screen(output) -> QScreen | None:
    """The QScreen a projector OutputSpec means: by serial, then name, then index."""
    screens = QGuiApplication.screens()
    if output.screen_serial:
        for s in screens:
            if s.serialNumber() == output.screen_serial:
                return s
    if output.screen_name:
        for s in screens:
            if s.name() == output.screen_name:
                return s
    if 0 <= output.screen_index < len(screens):
        return screens[output.screen_index]
    return None
