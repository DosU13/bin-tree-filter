"""Stitching cameras back into the stage."""

import os

import cv2
import numpy as np
import pytest

from core.compositor import Compositor
from core.scene import CameraSpec, ColorFix, SourceSpec, Stage, Transform
from tests.engine_cases import ROOT

STAGE = Stage(cols=3, rows=1, cell_size=540)
PROC = 64


def panorama() -> np.ndarray:
    return cv2.imread(os.path.join(ROOT, "scenes", "media", "panorama.jpg"))


def expected(pano: np.ndarray) -> np.ndarray:
    return cv2.resize(pano, (STAGE.cols * PROC, STAGE.rows * PROC), interpolation=cv2.INTER_AREA)


def crop_cameras(pano, rotation=0, mirror=False):
    cams, frames = [], {}
    for i, x0 in enumerate((0, 450, 900)):
        crop = pano[:, x0 : x0 + 720]
        img = cv2.flip(crop, 1) if mirror else crop
        # The frame the camera "sees" is the inverse of the display transform
        if rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if mirror and rotation in (90, 270):
            # mirror is applied before rotation: undo in the right order
            img = cv2.rotate(crop, {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: cv2.ROTATE_90_CLOCKWISE}[rotation])
            img = cv2.flip(img, 1)
        cam = CameraSpec(id=f"c{i}", source=SourceSpec(kind="video", width=img.shape[1], height=img.shape[0]),
                         transform=Transform(x=x0, y=0, width=720, height=540, rotation=rotation, mirror=mirror),
                         feather=120)
        cams.append(cam)
        frames[cam.id] = img
    return cams, frames


def test_three_overlapping_cameras_rebuild_the_panorama():
    pano = panorama()
    comp = Compositor(STAGE, PROC)
    cams, frames = crop_cameras(pano)
    img, covered = comp.compose(cams, frames)
    assert covered.all()
    err = np.abs(img.astype(int) - expected(pano).astype(int))
    assert err.mean() < 3.0


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_rotated_and_mirrored_cameras(rotation, mirror):
    pano = panorama()
    cams, frames = crop_cameras(pano, rotation, mirror)
    img, covered = Compositor(STAGE, PROC).compose(cams, frames)
    assert covered.all()
    assert np.abs(img.astype(int) - expected(pano).astype(int)).mean() < 3.0


def test_missing_camera_leaves_a_hole():
    pano = panorama()
    cams, frames = crop_cameras(pano)
    del frames["c1"]
    _, covered = Compositor(STAGE, PROC).compose(cams, frames)
    # Middle camera alone covers x 720..900 world px -> proc 85..106
    assert not covered[:, 86:106].any()
    assert covered[:, :80].all() and covered[:, 110:].all()


def test_colour_fix_matches_cameras():
    pano = panorama()
    cams, frames = crop_cameras(pano)
    frames["c2"] = cv2.convertScaleAbs(frames["c2"], alpha=1.0, beta=20)
    raw, _ = Compositor(STAGE, PROC).compose(cams, frames)
    cams[2].color = ColorFix(brightness=-20)
    fixed, _ = Compositor(STAGE, PROC).compose(cams, frames)
    ref = expected(pano).astype(int)
    assert np.abs(fixed.astype(int) - ref).mean() < np.abs(raw.astype(int) - ref).mean() - 2


def test_depth_mode_stitches_depth_and_skips_missing_readings():
    gray = np.full((540, 720), 30, np.uint8)  # far wall
    gray[:, 300:420] = 200                    # someone near
    gray[::3, 300:420] = 0                    # a third of their readings missing
    gray[:, 600:] = 0                         # nothing at all
    cam = CameraSpec(id="d", source=SourceSpec(width=720, height=540), transform=Transform(0, 0, 720, 540))
    depth, valid, covered = Compositor(STAGE, PROC).compose_depth([cam], {"d": gray})
    assert covered[:, :84].all() and not covered[:, 90:].any()
    assert np.allclose(depth[:, 40:44], 200, atol=0.5)          # holes don't pull it towards far
    assert np.allclose(depth[:, 10:30], 30, atol=0.5)
    assert valid[:, 40:44].all() and not valid[:, 72:84].any() and not valid[:, 90:].any()
    assert (depth[~valid] == 0).all()


def test_maps_follow_frame_size_changes():
    pano = panorama()
    cams, frames = crop_cameras(pano)
    comp = Compositor(STAGE, PROC)
    comp.compose(cams, frames)
    frames = {k: cv2.resize(v, (360, 270)) for k, v in frames.items()}
    img, covered = comp.compose(cams, frames)
    assert comp.maps["c0"].frame_size == (360, 270)
    assert np.abs(img.astype(int) - expected(pano).astype(int)).mean() < 4.0
