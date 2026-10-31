import time

import numpy as np
import pytest

import core.sources
from core import astra
from core.pipeline import Pipeline
from core.scene import CameraSpec, Scene, SceneError, SourceSpec, Stage, Transform
from core.sources import FrameReader, SourceError, make_reader


def astra_scene(**source) -> Scene:
    cam = CameraSpec(id="cam1", source=SourceSpec(kind="astra", width=640, height=480, **source),
                     transform=Transform(0, 0, 1440, 1080))
    return Scene(stage=Stage(cols=1, rows=1, cell_size=1080), cameras=[cam])


# ---------------------------------------------------------------------------
# Without hardware
# ---------------------------------------------------------------------------

def test_depth_maps_near_to_white_and_far_to_dark():
    mm = np.array([[0, 300, 500, 2500, 4500, 9000]], np.uint16)
    g = astra.depth_to_gray(mm, 0.5, 4.5)
    assert g.dtype == np.uint8
    assert g.tolist() == [[0, 255, 255, 128, 1, 1]]       # 0 = no reading, only
    ramp = astra.depth_to_gray(np.arange(500, 4501, 10, dtype=np.uint16), 0.5, 4.5)
    assert np.all(np.diff(ramp.astype(int)) <= 0) and ramp.min() == 1


def test_cleaner_removes_speckle_and_fills_shadows_from_the_far_side():
    mm = np.full((60, 640), 3000, np.uint16)         # a wall at 3 m
    mm[:, 100:200] = 1500                            # someone at 1.5 m
    mm[:, 200:230] = 0                               # their shadow on the wall
    mm[:, 400:460] = 0                               # a hole too wide to guess
    mm[:, 600:] = 0                                  # out of range up to the frame's edge
    mm[10, 300] = 0                                  # speckle: a lost reading
    mm[40, 320] = 800                                # speckle: a stray one
    out = astra.DepthCleaner()(mm, 0.0)
    assert (out[:, 100:200] == 1500).all()
    assert (out[:, 200:230] == 3000).all()           # the wall's depth, not the person's
    assert (out[:, 405:455] == 0).all() and (out[:, 605:] == 0).all()
    assert out[10, 300] == 3000 and out[40, 320] == 3000


def test_cleaner_smooths_noise_takes_movement_and_holds_dropouts():
    clean = astra.DepthCleaner()
    wall = np.full((20, 20), 3000, np.uint16)
    clean(wall, 0.0)
    assert 3000 < clean(wall + 40, 1 / 30)[10, 10] < 3040     # noise (within 3 % at 3 m): smoothed
    near = np.full_like(wall, 1500)
    assert (clean(near, 2 / 30) == 1500).all()                # someone stepped in: at once
    blank = np.zeros_like(wall)
    assert (clean(blank, 3 / 30) == 1500).all()               # readings lost for a moment: held
    assert (clean(blank, 3 / 30 + astra.HOLD_SECONDS) == 0).all()


def test_runtime_dir_lookup_order(tmp_path):
    def sdk(path):
        path.mkdir(parents=True)
        (path / "OpenNI2.dll").write_bytes(b"")
        return path

    project = tmp_path / "bin-tree-filter"
    project.mkdir()
    assert astra.runtime_dir(env={}, root=str(project)) is None
    sibling = sdk(tmp_path / "AstraSDK-v2.1.3" / "AstraSDK-v2.1.3" / "bin")
    assert astra.runtime_dir(env={}, root=str(project)) == str(sibling)
    vendor = sdk(project / "vendor" / "openni2")
    assert astra.runtime_dir(env={}, root=str(project)) == str(vendor)
    env = sdk(tmp_path / "elsewhere")
    assert astra.runtime_dir(env={"OPENNI2_REDIST64": str(env)}, root=str(project)) == str(env)
    assert astra.runtime_dir(env={"OPENNI2_REDIST64": str(tmp_path / "gone")}, root=str(project)) == str(vendor)


def test_scene_round_trips_an_astra_and_checks_it():
    scene = astra_scene(device_name="17112510331", stream="depth", depth_near=0.8, depth_far=3.0, depth_clean=False)
    again = Scene.from_dict(scene.to_dict())
    assert again.cameras[0].source == scene.cameras[0].source
    assert again.cameras[0].source.label() == "Astra 0 depth (17112510331)"
    for bad in ({"stream": "ir"}, {"depth_near": 3.0, "depth_far": 2.0}):
        with pytest.raises(SceneError):
            astra_scene(**bad).validate()


def test_missing_runtime_is_a_source_error(monkeypatch):
    monkeypatch.setattr(astra, "_ni", None)
    monkeypatch.setattr(astra, "runtime_dir", lambda *a, **k: None)
    reader = make_reader(SourceSpec(kind="astra"))
    assert isinstance(reader, astra.AstraReader)
    with pytest.raises(SourceError, match="OpenNI2 not found"):
        reader.open()


class RecordingReader(FrameReader):
    opened = []

    def __init__(self, spec):
        self.spec = spec
        RecordingReader.opened.append(self)

    def open(self):
        pass

    def read(self):
        return np.zeros((480, 640, 3), np.uint8)

    def update(self, spec):
        self.spec = spec


def test_depth_range_changes_without_reopening(monkeypatch):
    RecordingReader.opened = []
    monkeypatch.setattr(core.sources, "make_reader", lambda spec, resolve=None: RecordingReader(spec))
    scene = astra_scene()
    pipe = Pipeline(scene, realtime=False)
    try:
        src = pipe.sources["cam1"]
        scene.cameras[0].source.depth_far = 2.5
        scene.cameras[0].source.depth_clean = False
        pipe.set_scene(scene)
        assert pipe.sources["cam1"] is src and len(RecordingReader.opened) == 1
        assert src.reader.spec.depth_far == 2.5 and not src.reader.spec.depth_clean
        scene.cameras[0].source.stream = "colour"
        pipe.set_scene(scene)
        assert pipe.sources["cam1"] is not src and len(RecordingReader.opened) == 2
        assert src.status == "stopped"
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# With an Astra plugged in
# ---------------------------------------------------------------------------

def _need_astra():
    try:
        if not astra.list_astras():
            pytest.skip("no Astra connected")
    except SourceError as exc:
        pytest.skip(str(exc))


def _first_frame(reader, timeout=10.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        frame = reader.read()
        if frame is not None:
            return frame
    pytest.fail("the Astra delivered no frames")


@pytest.mark.astra
@pytest.mark.slow
def test_astra_depth_and_colour_share_one_device():
    _need_astra()
    depth = astra.AstraReader(SourceSpec(kind="astra", stream="depth", width=640, height=480))
    colour = astra.AstraReader(SourceSpec(kind="astra", stream="colour", width=640, height=480))
    depth.open()
    try:
        colour.open()
        try:
            assert len(astra._devices) == 1
            frame = _first_frame(depth)
            assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
            assert np.array_equal(frame[:, :, 0], frame[:, :, 2])          # grey
            assert depth.serial
            img = _first_frame(colour)
            assert img.shape == (480, 640, 3)
        finally:
            colour.close()
    finally:
        depth.close()
    assert astra._devices == {}
