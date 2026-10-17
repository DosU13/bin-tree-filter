import pytest

pytest.importorskip("PySide6")

from align_perform.canvas.handles import resize_rect
from align_perform.canvas.snapping import Snapper, targets_for
from core.scene import CameraSpec, Scene, Stage, Transform

R = (100.0, 50.0, 160.0, 90.0)      # 16:9


def test_corner_resize_keeps_opposite_corner_and_aspect():
    x, y, w, h = resize_rect(R, "br", (420, 200), aspect=16 / 9)
    assert (x, y) == (100, 50)
    assert w / h == pytest.approx(16 / 9)
    assert w == pytest.approx(320)


def test_top_left_corner_moves_origin():
    x, y, w, h = resize_rect(R, "tl", (180, 95), aspect=16 / 9)
    assert x + w == pytest.approx(260) and y + h == pytest.approx(140)
    assert w / h == pytest.approx(16 / 9)


def test_edge_resize_free_and_locked():
    x, y, w, h = resize_rect(R, "r", (300, 0))
    assert (x, y, w, h) == (100, 50, 200, 90)
    x, y, w, h = resize_rect(R, "r", (300, 0), aspect=16 / 9)
    assert w == 200 and h == pytest.approx(112.5)
    assert y + h / 2 == pytest.approx(95)          # stays vertically centred


def test_from_centre_and_minimum():
    x, y, w, h = resize_rect(R, "br", (300, 155), from_centre=True)
    assert x + w / 2 == pytest.approx(180) and y + h / 2 == pytest.approx(95)
    assert w == pytest.approx(240) and h == pytest.approx(120)
    x, y, w, h = resize_rect(R, "br", (0, 0))
    assert w == 8 and h == 8


def test_snapper_picks_nearest_target():
    s = Snapper(xs=[0, 500], ys=[0, 300], threshold=10)
    x, y, gx, gy = s.snap_move(493, 4, 100, 50)     # left edge near 500 -> no (493 is 7 away)
    assert x == 500 and y == 0 and gx == [500] and gy == [0]
    x, y, gx, gy = s.snap_move(395, 120, 100, 50)   # right edge 495 near 500
    assert x == 400 and gy == []
    s.enabled = False
    assert s.snap_move(493, 4, 100, 50)[:2] == (493, 4)


def test_targets_include_stage_cells_and_items():
    scene = Scene(stage=Stage(cols=3, rows=1, cell_size=100),
                  cameras=[CameraSpec(id="c", transform=Transform(10, 20, 50, 40))])
    xs, ys = targets_for(scene)
    assert {0, 100, 200, 300, 10, 60} <= set(xs)
    xs, _ = targets_for(scene, exclude=("camera", "c"))
    assert 10 not in xs
