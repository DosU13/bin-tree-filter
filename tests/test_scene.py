import json
import os

import numpy as np
import pytest

from core import params as P
from core.scene import CameraSpec, ProjectorSpec, Scene, SceneError, Stage, Transform
from tests.engine_cases import ROOT

EXAMPLE = os.path.join(ROOT, "scenes", "example.json")


def test_example_round_trips(tmp_path):
    scene = Scene.load(EXAMPLE)
    assert len(scene.cameras) == 3 and len(scene.projectors) == 2
    out = tmp_path / "copy.json"
    scene.save(str(out))
    again = Scene.load(str(out))
    assert again.stage == scene.stage
    assert [c.transform for c in again.cameras] == [c.transform for c in scene.cameras]
    assert again.params == scene.params
    assert again.looks == scene.looks


def test_media_paths_resolve_relative_to_the_scene():
    scene = Scene.load(EXAMPLE)
    path = scene.resolve(scene.cameras[0].source.path)
    assert os.path.isabs(path) and os.path.exists(path)


def test_saving_elsewhere_keeps_media_reachable(tmp_path):
    scene = Scene.load(EXAMPLE)
    for c in scene.cameras:
        c.source.path = scene.resolve(c.source.path)      # absolute, as the UI stores picks
    out = tmp_path / "moved.json"
    scene.save(str(out))
    again = Scene.load(str(out))
    assert os.path.exists(again.resolve(again.cameras[0].source.path))


def test_unknown_fields_are_kept(tmp_path):
    data = json.load(open(EXAMPLE, encoding="utf-8"))
    data["future_feature"] = {"x": 1}
    scene = Scene.from_dict(data)
    assert scene.to_dict()["future_feature"] == {"x": 1}


def test_auto_feather_serialises_as_auto():
    scene = Scene(projectors=[ProjectorSpec(id="p1")])
    assert scene.to_dict()["projectors"][0]["blend"]["feather"] == "auto"
    again = Scene.from_dict(scene.to_dict())
    assert again.projectors[0].blend.feather is None


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["stage"].update(cols=0), "cells"),
    (lambda d: d["cameras"][0].update(id="proj1"), "unique"),
    (lambda d: d["cameras"][0]["source"].update(kind="radio"), "source kind"),
    (lambda d: d["cameras"][0]["transform"].update(rotation=45), "rotation"),
    (lambda d: d["projectors"][0]["output"].update(width=0), "size"),
    (lambda d: d.update(version=99), "newer"),
])
def test_validation(mutate, message):
    data = json.load(open(EXAMPLE, encoding="utf-8"))
    mutate(data)
    with pytest.raises(SceneError, match=message):
        Scene.from_dict(data)


def test_bad_params_fall_back_to_defaults():
    scene = Scene.from_dict({"params": {"max_depth": "lots", "threshold": 0.5}})
    assert scene.params["max_depth"] == P.BY_KEY["max_depth"].default
    assert scene.params["threshold"] == 0.5


def test_new_id_and_lookup():
    scene = Scene(cameras=[CameraSpec(id="cam1")], projectors=[ProjectorSpec(id="proj1")])
    assert scene.new_id("cam") == "cam2"
    assert scene.item("camera", "cam1").id == "cam1"
    with pytest.raises(KeyError):
        scene.projector("nope")


def test_transform_quad_and_matrix():
    t = Transform(x=10, y=20, width=100, height=50)
    assert np.allclose(t.quad(), [[10, 20], [110, 20], [110, 70], [10, 70]])
    corners = [[0, 0], [100, 10], [90, 60], [-5, 50]]
    t.corners = corners
    m = t.norm_to_world()
    from core import geometry as geo
    assert np.allclose(geo.apply(m, [[0, 0], [1, 0], [1, 1], [0, 1]]), corners, atol=1e-6)
    assert t.bounds() == (-5.0, 0.0, 105.0, 60.0)


def test_stage_matrices_are_inverse():
    st = Stage(x=100, y=-50, cols=3, rows=2, cell_size=400)
    assert np.allclose(st.world_to_cell() @ st.cell_to_world(), np.eye(3))
    assert st.width == 1200 and st.height == 800
