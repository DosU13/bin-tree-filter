"""perform.py: runs, survives faults, keys, watchdog (opens real windows)."""

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("PySide6")

from tests.engine_cases import ROOT

pytestmark = [pytest.mark.gui, pytest.mark.gpu]
EXAMPLE = os.path.join(ROOT, "scenes", "example.json")


def run_perform(tmp_path, *extra, timeout=90):
    log_dir = str(tmp_path / "logs")
    cmd = [sys.executable, "-B", os.path.join(ROOT, "perform.py"), EXAMPLE, "--windowed", "--log-dir", log_dir, *extra]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    logs = ""
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            if name.endswith(".log"):
                logs += open(os.path.join(log_dir, name), encoding="utf-8").read()
    return proc.returncode, logs


@pytest.mark.slow
def test_runs_and_quits_cleanly(tmp_path):
    code, logs = run_perform(tmp_path, "--duration", "5")
    assert code == 0, logs[-2000:]
    assert "show started" in logs and "quitting" in logs
    assert "'proj1': 'shown', 'proj2': 'shown'" in logs


@pytest.mark.slow
def test_survives_injected_faults(tmp_path):
    code, logs = run_perform(tmp_path, "--duration", "9", "--simulate-faults", "0.8")
    assert code == 0, logs[-2000:]
    assert logs.count("FAULT") >= 5
    assert "reconnecting" in logs                     # cameras came back
    assert "injected fault" in logs                   # a step failed and the loop went on
    assert "quitting" in logs


def test_unknown_look_is_an_error(tmp_path):
    code, logs = run_perform(tmp_path, "--look", "nope", "--duration", "2")
    assert code == 2 and "not in scene" in logs


# -- in process ---------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from align_perform.app import make_app
    return make_app()


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def show(app, tmp_path):
    import perform
    args = perform.parse_args([EXAMPLE, "--windowed", "--fade", "0.3", "--blackout-fade", "0.2",
                               "--log-dir", str(tmp_path / "logs")])
    s = perform.Show(args)
    pump(app, 1.0)
    yield s
    s.quit()
    pump(app, 0.3)


def key(text, k=None, mods=None):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    if k is None:
        k = Qt.Key(ord(text.upper())) if text.isalpha() else Qt.Key(ord(text))
    return QKeyEvent(QEvent.Type.KeyPress, k, mods or Qt.KeyboardModifier.NoModifier, text)


def test_keys_change_the_live_look(app, show):
    before = show.params["max_depth"]
    show.on_key(key("D"))
    assert show.params["max_depth"] == before + 1
    assert show.session.scene.params["max_depth"] == before + 1    # running, not saved
    show.on_key(key("g"))
    assert show.params["cord"] is True


def test_number_keys_crossfade_to_looks(app, show):
    show.on_key(key("2"))                 # second look: "busy"
    pump(app, 0.6)
    assert show.look_name == "busy"
    assert show.params["max_depth"] == show.scene.looks["busy"]["max_depth"]
    show.on_key(key("0"))
    pump(app, 0.6)
    assert show.params["max_depth"] == show.base_params["max_depth"]


def test_blackout_freeze_and_escape(app, show):
    from PySide6.QtCore import Qt
    show.on_key(key("B"))
    pump(app, 0.5)
    assert show.outputs.level == 0.0
    show.on_key(key("B"))
    show.on_key(key("F"))
    pump(app, 0.15)                       # a frame already in progress may still land
    v = show.session.runner.latest().snapshot.version
    pump(app, 0.4)
    assert show.session.runner.latest().snapshot.version == v
    show.on_key(key("F"))
    show.on_key(key("", Qt.Key.Key_Escape))                         # ignored
    assert show.outputs.visible


def test_watchdog_restarts_a_dead_pipeline(app, show):
    old = show.session
    old.runner.stop()                                   # simulate the pipeline thread dying
    pump(app, 2.5)
    assert show.restarts == 1 and show.session is not old
    v = show.session.runner.latest()
    pump(app, 0.5)
    assert show.session.runner.latest() is not None
    assert v is None or show.session.runner.latest().snapshot.version > v.snapshot.version


def test_reload_picks_up_the_file(app, show):
    show.on_key(key("D"))
    show.on_key(key("R"))
    assert show.params == show.base_params


def test_missing_display_waits(app, tmp_path):
    from align_perform.runtime.outputs import ProjectorOutputs
    from core.scene import Scene
    scene = Scene.load(EXAMPLE)
    for p in scene.projectors:
        p.output.kind, p.output.screen_name, p.output.screen_index = "screen", "NO-SUCH-DISPLAY", 99

    class Dummy:
        tile = None
        last = None

    outs = ProjectorOutputs(scene, Dummy(), missing="wait")
    outs.set_visible(True)
    pump(app, 0.3)
    assert all(w.state == "missing" and not w.isVisible() for w in outs.windows.values())
    outs.close()
