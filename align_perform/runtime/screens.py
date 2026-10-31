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


def pick_screen(output, keys: list[tuple[str, str]], by_index: bool = True) -> int | None:
    """Which display a projector OutputSpec means: by serial, then name, then index.

    Args:
        output: the projector's OutputSpec.
        keys: (serial, name) of each connected display, in screen order.
        by_index: fall back to the saved index when no serial or name matches.

    Returns:
        Index into *keys*, or None.

    Projectors of one make often report the same placeholder serial and name (Epson:
    16843009, "EPSON PJ"), so when several displays match, the saved index picks one.
    """
    index = output.screen_index if 0 <= output.screen_index < len(keys) else None
    for field, value in enumerate((output.screen_serial, output.screen_name)):
        if not value:
            continue
        hits = [i for i, k in enumerate(keys) if k[field] == value]
        if hits:
            return index if index in hits else hits[0]
    return index if by_index else None


def find_screen(output) -> QScreen | None:
    """The QScreen a projector OutputSpec means (see pick_screen)."""
    screens = QGuiApplication.screens()
    i = pick_screen(output, [(s.serialNumber(), s.name()) for s in screens])
    return None if i is None else screens[i]
