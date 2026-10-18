"""Resize handles: where they are, which cursor they show, and the resize maths.

Rectangles are (x, y, w, h) in world px.  Handle names: tl, t, tr, r, br, b, bl, l.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

HANDLES = ("tl", "t", "tr", "r", "br", "b", "bl", "l")
CORNERS = ("tl", "tr", "br", "bl")          # corner-pin handle order = quad order
HANDLE_PX = 9                               # handle size on screen

# (sx, sy): which edges a handle moves (-1 left/top, +1 right/bottom, 0 none)
SIDES = {"tl": (-1, -1), "t": (0, -1), "tr": (1, -1), "r": (1, 0),
         "br": (1, 1), "b": (0, 1), "bl": (-1, 1), "l": (-1, 0)}

CURSORS = {
    "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
    "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
    "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
}


def handle_points(x: float, y: float, w: float, h: float) -> dict[str, tuple[float, float]]:
    cx, cy = x + w / 2, y + h / 2
    return {"tl": (x, y), "t": (cx, y), "tr": (x + w, y), "r": (x + w, cy),
            "br": (x + w, y + h), "b": (cx, y + h), "bl": (x, y + h), "l": (x, cy)}


def resize_rect(rect: tuple[float, float, float, float], handle: str, p: tuple[float, float],
                aspect: float | None = None, from_centre: bool = False,
                min_size: float = 8.0) -> tuple[float, float, float, float]:
    """New rectangle after dragging *handle* to world point *p*.

    Args:
        aspect:      Keep width / height at this ratio (None = free).
        from_centre: Resize symmetrically around the centre (Alt).
    """
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    sx, sy = SIDES[handle]
    k = 2.0 if from_centre else 1.0

    w2, h2 = w, h
    if sx:
        ax = cx if from_centre else (x + w if sx < 0 else x)
        w2 = max(min_size, (p[0] - ax) * sx * k)
    if sy:
        ay = cy if from_centre else (y + h if sy < 0 else y)
        h2 = max(min_size, (p[1] - ay) * sy * k)

    if aspect:
        if sx and sy:
            # Corner: follow whichever axis moved further, keep the ratio
            if w2 / w >= h2 / h:
                h2 = w2 / aspect
            else:
                w2 = h2 * aspect
        elif sx:
            h2 = w2 / aspect
        else:
            w2 = h2 * aspect

    # The fixed side stays put; an axis the handle doesn't drag stays centred
    if from_centre or sx == 0:
        x2 = cx - w2 / 2
    else:
        x2 = x if sx > 0 else x + w - w2
    if from_centre or sy == 0:
        y2 = cy - h2 / 2
    else:
        y2 = y if sy > 0 else y + h - h2
    return x2, y2, w2, h2
