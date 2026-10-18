import cv2
import numpy as np
import pytest

from core import geometry as geo


def test_rect_and_inverse():
    m = geo.rect_matrix(10, 20, 200, 100)
    pts = geo.apply(m, [[0, 0], [1, 1], [0.5, 0.25]])
    assert np.allclose(pts, [[10, 20], [210, 120], [110, 45]])
    assert np.allclose(geo.apply(geo.invert(m), pts), [[0, 0], [1, 1], [0.5, 0.25]])


def test_quad_distance_sign_and_winding():
    quad = [[0, 0], [10, 0], [10, 5], [0, 5]]
    for q in (quad, quad[::-1]):
        d = geo.quad_distance([[5, 2.5], [1, 1], [-1, 2], [5, 6]], q)
        assert np.allclose(d, [2.5, 1, -1, -1])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_source_to_display_matches_opencv(rotation, mirror):
    """The matrix agrees with actually flipping / rotating an image."""
    w, h = 8, 5
    src = np.arange(w * h, dtype=np.float32).reshape(h, w)
    img = cv2.flip(src, 1) if mirror else src
    rot = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
    disp = img if rot is None else cv2.rotate(img, rot)
    dw, dh = geo.display_size(w, h, rotation)
    assert disp.shape == (dh, dw)
    m = geo.source_to_display(w, h, rotation, mirror)
    for sy in range(h):
        for sx in range(w):
            u, v = geo.apply(m, [[sx + 0.5, sy + 0.5]])[0]
            dx, dy = int(u * dw), int(v * dh)
            assert disp[dy, dx] == src[sy, sx]
