import os
import time

import numpy as np
import pytest

from core import geometry as geo
from core.pipeline import Pipeline, PipelineRunner
from core.scene import Scene
from core.view import blend_alpha, blend_spec, projector_view, stage_view
from tests.engine_cases import ROOT

EXAMPLE = os.path.join(ROOT, "scenes", "example.json")


@pytest.fixture
def scene():
    return Scene.load(EXAMPLE)


def test_steps_produce_a_tree_for_the_stage(scene):
    pipe = Pipeline(scene, realtime=False)
    try:
        for _ in range(5):
            r = pipe.step(1 / 15)
        st = scene.stage
        assert r.snapshot.cols == st.cols and r.snapshot.rows == st.rows
        assert r.stage_input.shape == (st.rows * 128, st.cols * 128, 3)
        assert r.covered.all()
        assert r.leaves > 3 * 2
        assert set(r.camera_status.values()) == {"ok"}
        assert r.snapshot.version == 5
    finally:
        pipe.close()


def test_uncovered_stage_is_plain_background(scene):
    scene.cameras = scene.cameras[:1]            # only the left third is seen
    pipe = Pipeline(scene, realtime=False)
    try:
        for _ in range(3):
            r = pipe.step(1 / 15)
        bg = np.array([253, 217, 181])[::-1]      # params background, BGR
        assert not r.covered[:, 200:].any()
        assert np.all(r.labels_image[:, 200:] == bg)
    finally:
        pipe.close()


def test_param_and_scene_changes(scene):
    pipe = Pipeline(scene, realtime=False)
    try:
        pipe.step(1 / 15)
        pipe.set_params(dict(pipe.params, proc_size=64, colors=3))
        r = pipe.step(1 / 15)
        assert r.stage_input.shape == (64, 192, 3)
        assert pipe.tracker.k == 3
        moved = pipe.scene.copy()
        moved.cameras[0].transform.x += 30
        pipe.set_scene(moved)
        assert "cam1" not in pipe.compositor.maps      # rebuilt lazily
        pipe.step(1 / 15)
        assert "cam1" in pipe.compositor.maps
        moved = pipe.scene.copy()
        moved.stage.cols = 4
        pipe.set_scene(moved)
        assert pipe.tree.cols == 4
        assert pipe.step(1 / 15).snapshot.cols == 4
    finally:
        pipe.close()


def test_depth_mode(scene):
    scene.params["mode"] = "depth"
    pipe = Pipeline(scene, realtime=False)
    try:
        r = pipe.step(1 / 15)
        assert len(np.unique(r.labels_image.reshape(-1, 3), axis=0)) <= 2
    finally:
        pipe.close()


def test_runner_keeps_going_after_errors(scene, monkeypatch):
    real_step = Pipeline.step
    calls = {"n": 0}

    def flaky_step(self, dt):
        calls["n"] += 1
        if calls["n"] in (3, 4):
            raise RuntimeError("simulated failure")
        return real_step(self, dt)

    monkeypatch.setattr(Pipeline, "step", flaky_step)
    runner = PipelineRunner(scene, max_fps=60).start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and (runner.latest() is None or runner.latest().snapshot.version < 10):
            time.sleep(0.05)
        assert runner.latest().snapshot.version >= 10
        assert runner.errors == 2 and "simulated failure" in runner.last_error
        runner.set_params(dict(scene.params, max_depth=5))
        time.sleep(0.8)
        assert runner.fps > 5
    finally:
        runner.stop()


def test_projector_view_maps_corners_to_cells(scene):
    proj = scene.projector("proj2")
    view = projector_view(scene, proj, 192, 108)
    corners = geo.apply(view.norm_to_cell, [[0, 0], [1, 1]])
    assert np.allclose(corners, [[660 / 540, 0], [1620 / 540, 1]])
    assert view.width == 192 and view.gain == 1.0


def test_auto_blend_feather_is_the_overlap(scene):
    spec = blend_spec(scene, scene.projector("proj1"))
    assert spec is not None and spec.index == 0
    assert np.allclose(spec.feather, 300 / 540)
    solo = scene.copy()
    solo.projectors[1].transform.x = 2000
    assert blend_spec(solo, solo.projectors[0]) is None


def test_blend_weights_sum_to_one_in_light(scene):
    spec0 = blend_spec(scene, scene.projectors[0])
    spec1 = blend_spec(scene, scene.projectors[1])
    xs = np.linspace(0.01, 2.99, 300)
    pts = np.stack([xs, np.full_like(xs, 0.5)], axis=1)
    light = blend_alpha(pts, spec0) ** spec0.gamma + blend_alpha(pts, spec1) ** spec1.gamma
    assert np.allclose(light, 1.0, atol=1e-3)


def test_stage_view_aspect(scene):
    v = stage_view(scene, 300)
    assert (v.width, v.height) == (300, 100)


def test_blend_ramp_is_purely_across_the_overlap(scene):
    """Outer stage borders don't fade: the ramp near the top edge equals the one in the middle."""
    spec = blend_spec(scene, scene.projectors[0])
    assert spec.edges[0].tolist() == [False, True, False, False]   # only the left projector's right edge
    xs = np.linspace(660 / 540 + 0.005, 960 / 540 - 0.005, 11)    # across the overlap
    top = blend_alpha(np.stack([xs, np.full_like(xs, 0.01)], axis=1), spec)
    mid = blend_alpha(np.stack([xs, np.full_like(xs, 0.5)], axis=1), spec)
    assert np.allclose(top, mid, atol=2e-3)
    assert top[0] > 0.95 and top[-1] < 0.1


# ---------------------------------------------------------------------------
# Presence, waves, idle look (scenes/people.json: two walkers, then nobody)
# ---------------------------------------------------------------------------

PEOPLE = os.path.join(ROOT, "scenes", "people.json")


def test_people_scene_finds_walkers_sends_waves_and_goes_idle():
    scene = Scene.load(PEOPLE)
    pipe = Pipeline(scene, realtime=False)
    try:
        most, rings, idle_peak = 0, 0, 0.0
        for i in range(int(17.5 * 15)):
            r = pipe.step(1 / 15)
            if i == 20:                                  # 1.4 s: background learned, nobody yet
                assert not r.learning and r.people_count == 0 and r.activity.max() == 0
            most = max(most, r.people_count)
            if r.snapshot.waves is not None:
                rings = max(rings, len(r.snapshot.waves))
            idle_peak = max(idle_peak, r.idle)
            if i == 6 * 15:                              # 6 s: walkers on, show look in force
                assert r.idle == 0.0 and pipe.effective["presence"]
        assert most == 2 and rings >= 2
        assert idle_peak == 1.0 and r.idle == 1.0        # nobody since 10.5 s, idle after 4 s
        assert pipe.effective["max_depth"] == scene.looks["idle"]["max_depth"]
        assert not pipe.effective["presence"]
        assert pipe.effective["presence_threshold"] == pipe.params["presence_threshold"]   # show setting
    finally:
        pipe.close()


def test_presence_off_costs_nothing(scene):
    pipe = Pipeline(scene, realtime=False)
    try:
        r = pipe.step(1 / 15)
        assert r.activity is None and r.people == [] and "presence" not in r.timings
        assert r.snapshot.waves is None
    finally:
        pipe.close()


def test_idle_needs_permission_and_relearn_on_request():
    scene = Scene.load(PEOPLE)
    pipe = Pipeline(scene, realtime=False)
    pipe.idle_enabled = False
    try:
        for _ in range(int(17.5 * 15)):
            r = pipe.step(1 / 15)
        assert r.idle == 0.0 and pipe.effective["presence"]
        pipe.relearn_background()
        assert pipe.step(1 / 15).learning
        moved = pipe.scene.copy()
        moved.cameras[0].transform.x += 10
        for _ in range(10):
            pipe.step(1 / 15)
        pipe.set_scene(moved)                            # the stage picture changes: relearn
        assert pipe.step(1 / 15).learning
    finally:
        pipe.close()
