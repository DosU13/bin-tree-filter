"""Drive live.py's per-frame pipeline without opening a window."""

import os

import cv2
import numpy as np

import live
from config import BACKGROUND_COLOR, ORNAMENT_FILENAME
from core.engine import OrnamentEngine, load_tile
from core.posterize import PALETTE_STYLES, PaletteTracker
from tests.engine_cases import ROOT


def make_engine(**kw):
    tile = load_tile(os.path.join(ROOT, ORNAMENT_FILENAME), crop=True)
    return OrnamentEngine(tile, out_size=128, proc_size=48, max_level=9, background=BACKGROUND_COLOR, **kw)


def test_colour_mode_on_still_image():
    engine = make_engine()
    tracker = PaletteTracker(k=4)
    src = live.open_source(os.path.join(ROOT, "me.jpg"))
    assert src.is_opened
    for i, style in enumerate(list(PALETTE_STYLES) * 2):
        tracker.style = style
        ok, frame = src.read()
        small = cv2.resize(live.centre_square(frame), (48, 48), interpolation=cv2.INTER_AREA)
        small = live.apply_sharpen(live.apply_blur(small, 1), 0.4)
        preview = live.apply_filters(small, 10, 1.2, 1.3)
        labels, palette = tracker.update(preview, 1 / 30)
        engine.update(labels, palette, 1 / 30)
        out = engine.render(cord=i % 2 == 0, wireframe=i % 3 == 0)
    assert out.shape == (128, 128, 3)
    assert engine.leaf_count > 2
    display, side = live.compose(out, False, 200)
    live.paste_thumb(display, palette[labels].astype(np.uint8), 120, 120, 40, "Labels")
    live.put_text(display, "ok", 10, 25)
    assert display.shape == (200, 200, 3) and side == 200


def test_depth_mode_on_gif():
    engine = make_engine()
    gif = live.open_source(os.path.join(ROOT, "depth.gif"))
    for _ in range(3):
        ok, frame = gif.read()
        gray = cv2.cvtColor(live.centre_square(frame), cv2.COLOR_BGR2GRAY)
        labels = live.depth_labels(gray, 60, 140, engine.proc_size)
        pal = np.array([live.BG_PRESETS[0], live.FG_PRESETS[0]], np.float32)
        engine.update(labels, pal, 1 / 30)
        out = engine.render()
    gif.release()
    assert out.shape == (128, 128, 3)
