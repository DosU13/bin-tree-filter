import cv2
import numpy as np

from core.posterize import PALETTE_STYLES, DepthLayers, PaletteTracker, depth_palette, split_main
from tests.engine_cases import ROOT


def small_image(size=64):
    img = cv2.imread(f"{ROOT}/me.jpg")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def test_identical_frames_give_identical_labels():
    cv2.setRNGSeed(1)
    tracker = PaletteTracker(k=4)
    img = small_image()
    tracker.update(img, 1 / 30)
    first, _ = tracker.update(img, 1 / 30)
    for _ in range(5):
        labels, palette = tracker.update(img, 1 / 30)
        assert np.array_equal(labels, first)
    assert labels.shape == (64, 64)
    assert palette.shape == (4, 3)


def test_noise_does_not_flicker_much():
    cv2.setRNGSeed(2)
    rng = np.random.default_rng(0)
    tracker = PaletteTracker(k=4)
    img = small_image().astype(np.float32)
    prev = None
    changed = []
    for _ in range(10):
        noisy = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        labels, _ = tracker.update(noisy, 1 / 30)
        if prev is not None:
            changed.append(np.mean(labels != prev))
        prev = labels
    assert max(changed[2:]) < 0.02


def test_styles_produce_k_colours():
    tracker = PaletteTracker(k=5)
    img = small_image()
    for style in PALETTE_STYLES:
        tracker.style = style
        _, palette = tracker.update(img, 1 / 30)
        assert palette.shape == (5, 3)
        assert palette.min() >= 0 and palette.max() <= 255


# ---------------------------------------------------------------------------
# Depth layers
# ---------------------------------------------------------------------------

def test_depth_layers_slice_the_band_nearest_first():
    depth = np.array([[250, 215, 190, 130, 100, 60, 20, 200]], np.float32)
    valid = np.array([[1, 1, 1, 1, 1, 1, 1, 0]], bool)
    labels = DepthLayers().update(depth, valid, 60, 220, 4)        # slices of 40 grey levels
    assert labels.tolist() == [[0, 1, 1, 3, 4, 4, 0, 0]]          # outside / no reading = 0
    band = DepthLayers().update(depth, valid, 60, 220, 1)
    assert band.tolist() == [[0, 1, 1, 1, 1, 1, 0, 0]]


def test_depth_layers_hold_their_label_near_an_edge():
    layers = DepthLayers()
    valid = np.ones((1, 3), bool)
    edge = 180.0                                                   # between slices 1 and 2 of 60..220 / 4
    assert layers.update(np.array([[edge + 1, edge - 1, 100]], np.float32), valid, 60, 220, 4).tolist() == [[1, 2, 4]]
    jitter = np.array([[edge - 2, edge + 2, 100]], np.float32)    # crossed, but by less than the margin
    assert layers.update(jitter, valid, 60, 220, 4).tolist() == [[1, 2, 4]]
    moved = np.array([[edge - 10, edge + 10, 100]], np.float32)
    assert layers.update(moved, valid, 60, 220, 4).tolist() == [[2, 1, 4]]
    # A new band starts afresh
    assert layers.update(jitter, valid, 60, 200, 4).tolist() == [[1, 1, 3]]


def test_depth_palette():
    fg, bg = (0, 0, 160), (245, 235, 210)
    fade = depth_palette(4, "fade", fg, bg)
    assert fade.shape == (5, 3)
    assert np.allclose(fade[0], bg) and np.allclose(fade[1], fg)
    dist = np.abs(fade[1:] - np.array(bg, np.float32)).sum(axis=1)
    assert np.all(np.diff(dist) < 0)                               # farther layers fade into the background
    # A palette's nearest layer is the one least like the background: dark on light, light on dark
    light = depth_palette(4, "tush-kiyiz", fg, bg)[1:].sum(axis=1)
    dark = depth_palette(4, "tush-kiyiz", fg, (20, 20, 20))[1:].sum(axis=1)
    assert np.all(np.diff(light) > 0) and np.all(np.diff(dark) < 0)
    assert depth_palette(1, "fade", fg, bg).tolist() == [list(bg), list(fg)]


def test_glow_keeps_the_palettes_main_colour_for_itself(monkeypatch):
    rest, light = split_main("tush-kiyiz")
    assert light == "#c1121f" and rest == ["#0b3c49", "#f4a259", "#fdf0d5"]
    assert split_main("navy-red-gold")[1] == "#9b2226"
    assert split_main("natural") is None and split_main("fade") is None
    red = np.array([0x1f, 0x12, 0xc1], np.float32)                   # BGR
    fg, bg = (0, 0, 160), (40, 20, 20)

    def has_red(palette):
        return bool((np.abs(palette - red).max(axis=1) < 3).any())

    assert has_red(depth_palette(4, "tush-kiyiz", fg, bg))
    assert not has_red(depth_palette(4, "tush-kiyiz", fg, bg, glow=True))
    tracker = PaletteTracker(k=4)
    tracker.style = "tush-kiyiz"
    _, palette = tracker.update(small_image(), 1 / 30)
    assert has_red(palette)
    tracker.glow = True
    assert not has_red(tracker.palette())
    # A style without a named main colour lends the glow its most saturated one
    monkeypatch.setitem(PALETTE_STYLES, "test", ["#202020", "#3060c0", "#d0d0d0"])
    assert split_main("test") == (["#202020", "#d0d0d0"], "#3060c0")
