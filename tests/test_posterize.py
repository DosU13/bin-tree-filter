import cv2
import numpy as np

from core.posterize import PALETTE_STYLES, PaletteTracker
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
