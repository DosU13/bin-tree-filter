"""Motion glow: light from movement and people, carried on by the fluid."""

import os

import numpy as np

from core.glow import LEVELS, MotionGlow, glow_labels
from core.pipeline import Pipeline
from core.scene import Scene
from tests.engine_cases import ROOT

CELL = 64          # proc px per cell in these tests: a 2 x 1 cell stage of 128 x 64 px
DT = 1 / 30


def room() -> np.ndarray:
    """A depth picture of an empty room: a far wall (dark) with a nearer floor."""
    img = np.full((CELL, 2 * CELL), 60, np.float32)
    img[48:] = 110
    return img


def with_box(img: np.ndarray, x: int, y: int = 16, size: int = 16, depth: float = 200) -> np.ndarray:
    out = img.copy()
    out[y : y + size, x : x + size] = depth
    return out


def run(glow: MotionGlow, frames, **kw) -> np.ndarray:
    valid = np.ones((CELL, 2 * CELL), bool)
    for f in frames:
        out = glow.update(f, valid, DT, cell_px=CELL, **kw)
    return out


def test_still_room_stays_dark_and_movement_lights_up():
    glow = MotionGlow()
    assert run(glow, [room()] * 20).max() == 0
    lit = run(glow, [with_box(room(), x) for x in range(20, 40, 4)], fade=0.5)
    assert lit[18:30, 38:54].mean() > 0.5               # where the box is now
    assert lit[2:12, 100:120].max() == 0                # far from it


def test_person_is_lit_against_the_learned_environment():
    glow = MotionGlow()
    # Someone wanders through a third of the time: the environment is still the room
    frames = [with_box(room(), 70) if i % 3 == 0 else room() for i in range(90)]
    run(glow, frames, source="person", flow=0, memory=30, fade=0.1)
    lit = run(glow, [with_box(room(), 20)] * 10, source="person", flow=0, memory=30, fade=0.1)
    assert lit[18:30, 22:34].min() == 1.0               # standing still, but lit
    assert lit[18:30, 72:84].max() < LEVELS[0]          # the place they passed through is not
    assert lit[:, 100:].max() < LEVELS[0]


def test_fluid_carries_the_light_on_after_the_movement_stops():
    def centre(img):
        w = np.clip(img - LEVELS[0], 0, None)
        return (w.sum(axis=0) * np.arange(img.shape[1])).sum() / max(w.sum(), 1e-6)

    def after(flow):
        glow = MotionGlow()
        run(glow, [room()] * 5)
        run(glow, [with_box(room(), x) for x in range(10, 50, 4)], flow=flow, fade=2.0)
        return run(glow, [room()] * 10, flow=flow, fade=2.0)   # the box is gone

    still, flowing = after(0.0), after(2.0)
    assert still.max() > LEVELS[0] and flowing.max() > LEVELS[0]
    assert centre(flowing) > centre(still) + 4         # the light kept moving right


def test_glow_labels_add_three_shades():
    labels = np.zeros((1, 4), np.int64)
    palette = np.array([[255, 255, 255], [0, 0, 0]], np.float32)
    glow = np.array([[0.0, 0.2, 0.5, 0.9]], np.float32)
    out, pal = glow_labels(labels, palette, glow, (0, 0, 255), (255, 255, 255))
    assert out.tolist() == [[0, 2, 3, 4]]
    assert pal.shape == (5, 3) and pal[4].tolist() == [0, 0, 255]
    assert np.allclose(pal[2], [170, 170, 255])         # a third of the way to the light colour


def test_pipeline_paints_moving_people_in_the_light_colour():
    scene = Scene.load(os.path.join(ROOT, "scenes", "people.json"))
    scene.params.update(glow=True, glow_color="#00ff00", presence=False, waves=False, idle_look="")
    pipe = Pipeline(scene, realtime=False)
    try:
        for _ in range(40):
            r = pipe.step(1 / 15)
        assert "glow" in r.timings
        green = (r.labels_image == (0, 255, 0)).all(axis=2)
        assert 0.001 < green.mean() < 0.3                # the walkers, not the whole stage
    finally:
        pipe.close()


def test_glow_takes_the_palettes_main_colour():
    from core import params as P
    from core.pipeline import glow_colour
    v = P.normalise({"mode": "depth", "depth_palette": "navy-red-gold", "glow": True, "glow_color": "#00ff00"})
    assert glow_colour(v) == (0x26, 0x22, 0x9b)                      # #9b2226, BGR
    assert glow_colour(dict(v, depth_palette="fade")) == (0, 255, 0)
    assert glow_colour(dict(v, mode="colour", palette_style="shyrdak")) == (0x1c, 0x2e, 0x7a)
