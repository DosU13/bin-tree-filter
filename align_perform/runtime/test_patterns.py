"""Overlay images for projector outputs (RGBA, drawn with OpenCV in the output's pixel space).

test_pattern     world grid (cell lines thicker), stage outline, other projectors'
                 overlap zones, camera outlines, corner tags, projector name
identify         big projector number
hud              a few lines of status text (perform.py)

The test pattern is drawn over a flat grey field rendered by the GPU with edge
blending applied, so uneven overlaps show up immediately.
"""

from __future__ import annotations

import cv2
import numpy as np

from core import geometry as geo

CORNER_NAMES = ("TL", "TR", "BR", "BL")


def world_to_pixel(proj, width: int, height: int) -> np.ndarray:
    """World -> this projector's output pixels."""
    return geo.translate_scale(0, 0, width, height) @ geo.invert(proj.transform.norm_to_world())


def _pts(m: np.ndarray, world_pts) -> np.ndarray:
    return np.round(geo.apply(m, world_pts)).astype(np.int32)


def _line(img, m, a, b, color, thickness) -> None:
    p = _pts(m, [a, b])
    cv2.line(img, tuple(p[0]), tuple(p[1]), color, thickness, cv2.LINE_AA)


def _text(img, text: str, org, scale: float, color=(255, 255, 255, 255), thick: int = 2, center: bool = False) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    if center:
        x, y = int(x - tw / 2), int(y + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def test_pattern(scene, proj, width: int, height: int, number: int = 1, corner: int | None = None) -> np.ndarray:
    """RGBA test pattern for *proj* at width x height (transparent where nothing is drawn)."""
    img = np.zeros((height, width, 4), np.uint8)
    m = world_to_pixel(proj, width, height)
    k = height / 1080
    st = scene.stage
    x0, y0, w, h = proj.transform.bounds()
    pad = max(w, h)
    xa, xb, ya, yb = x0 - pad, x0 + w + pad, y0 - pad, y0 + h + pad

    # Grid: minor lines every quarter cell, major lines on cell borders
    step = st.cell_size / 4
    i0, i1 = int(np.floor((xa - st.x) / step)), int(np.ceil((xb - st.x) / step))
    for i in range(i0, i1 + 1):
        x = st.x + i * step
        major = i % 4 == 0
        _line(img, m, (x, ya), (x, yb), (255, 255, 255, 230 if major else 110), max(1, round((3 if major else 1) * k)))
    j0, j1 = int(np.floor((ya - st.y) / step)), int(np.ceil((yb - st.y) / step))
    for j in range(j0, j1 + 1):
        y = st.y + j * step
        major = j % 4 == 0
        _line(img, m, (xa, y), (xb, y), (255, 255, 255, 230 if major else 110), max(1, round((3 if major else 1) * k)))

    # Stage outline
    q = _pts(m, st.quad())
    cv2.polylines(img, [q], True, (40, 230, 255, 255), max(2, round(5 * k)), cv2.LINE_AA)

    # Other projectors (their edges mark the blend zones) and cameras: outlines only,
    # so the blended grey field underneath stays easy to judge
    for other in scene.enabled_projectors():
        if other.id != proj.id:
            cv2.polylines(img, [_pts(m, other.transform.quad())], True, (40, 150, 255, 255),
                          max(2, round(3 * k)), cv2.LINE_AA)
    for cam in scene.enabled_cameras():
        cv2.polylines(img, [_pts(m, cam.transform.quad())], True, (255, 170, 80, 200),
                      max(1, round(2 * k)), cv2.LINE_AA)

    # Output border, centre cross, corner tags
    b = max(2, round(4 * k))
    cv2.rectangle(img, (b // 2, b // 2), (width - 1 - b // 2, height - 1 - b // 2), (60, 255, 60, 255), b)
    c = (width // 2, height // 2)
    r = round(60 * k)
    cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), (60, 255, 60, 255), b)
    cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r), (60, 255, 60, 255), b)
    inset = round(70 * k)
    for i, (px, py) in enumerate(((inset, inset), (width - inset, inset), (width - inset, height - inset),
                                  (inset, height - inset))):
        sel = corner == i
        color = (0, 220, 255, 255) if sel else (255, 255, 255, 255)
        if sel:
            cv2.circle(img, (px, py), round(45 * k), color, b, cv2.LINE_AA)
        _text(img, CORNER_NAMES[i], (px, py), 1.2 * k, color, max(2, round(3 * k)), center=True)

    title = f"{number}: {proj.name or proj.id}"
    _text(img, title, (width // 2, height // 2 - round(140 * k)), 2.6 * k, thick=max(2, round(5 * k)), center=True)
    sub = f"{proj.output.width} x {proj.output.height}"
    if proj.transform.corners is not None:
        sub += "   4-corner warp" + ("" if corner is None else f"   corner {CORNER_NAMES[corner]}")
    _text(img, sub, (width // 2, height // 2 + round(140 * k)), 1.3 * k, thick=max(2, round(3 * k)), center=True)
    _text(img, "arrows: move  Shift: x10  Tab: pick corner", (width // 2, height - round(110 * k)), 0.9 * k,
          (220, 220, 220, 255), max(1, round(2 * k)), center=True)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def identify(number: int, name: str, width: int, height: int) -> np.ndarray:
    """RGBA: the projector's number, huge, on a dark plate."""
    img = np.zeros((height, width, 4), np.uint8)
    k = height / 1080
    cv2.rectangle(img, (width // 2 - round(330 * k), height // 2 - round(330 * k)),
                  (width // 2 + round(330 * k), height // 2 + round(330 * k)), (0, 0, 0, 200), -1)
    _text(img, str(number), (width // 2, height // 2 - round(30 * k)), 14 * k, (255, 255, 255, 255),
          max(4, round(30 * k)), center=True)
    _text(img, name, (width // 2, height // 2 + round(250 * k)), 1.8 * k, thick=max(2, round(4 * k)), center=True)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def hud(lines: list[str], width: int, height: int) -> np.ndarray:
    """RGBA: status lines top-left on a translucent plate."""
    img = np.zeros((height, width, 4), np.uint8)
    k = max(0.5, height / 1080)
    lh = round(34 * k)
    cv2.rectangle(img, (0, 0), (round(900 * k), lh * len(lines) + round(16 * k)), (0, 0, 0, 160), -1)
    for i, line in enumerate(lines):
        _text(img, line, (round(14 * k), round(10 * k) + lh * (i + 1) - round(8 * k)), 0.8 * k,
              thick=max(1, round(2 * k)))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
