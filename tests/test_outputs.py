"""Projector output windows (opens real windows on the desktop)."""

import os
import time

import pytest

pytest.importorskip("PySide6")

from tests.engine_cases import ROOT

pytestmark = [pytest.mark.gui, pytest.mark.gpu]
EXAMPLE = os.path.join(ROOT, "scenes", "example.json")


@pytest.fixture(scope="module")
def app():
    from align_perform.app import make_app
    from PySide6.QtCore import QSettings
    a = make_app()
    QSettings().clear()
    return a


def pump(app, seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def win(app):
    from align_perform.main_window import MainWindow
    w = MainWindow(EXAMPLE)
    w.show()
    pump(app, 1.0)
    yield w
    w.doc.undo.setClean()
    w.close()
    pump(app, 0.3)


def grab(window):
    from align_perform.runtime.qt_gl import qimage_to_bgr
    return qimage_to_bgr(window.grabFramebuffer())


def test_outputs_open_render_and_close(app, win):
    win.actions["outputs"].setChecked(True)
    pump(app, 1.0)
    outs = win.outputs
    assert set(outs.windows) == {"proj1", "proj2"}
    assert all(w.isVisible() and w.frames > 5 for w in outs.windows.values())
    assert outs.states() == {"proj1": "shown", "proj2": "shown"}
    img = grab(outs.windows["proj1"])
    assert img.std() > 10                       # the ornament, not a flat colour
    win.actions["outputs"].setChecked(False)
    pump(app, 0.2)
    assert not any(w.isVisible() for w in outs.windows.values())


def test_test_pattern_and_blend_field(app, win):
    win.actions["test_pattern"].setChecked(True)   # also opens the outputs
    pump(app, 0.8)
    outs = win.outputs
    a = grab(outs.windows["proj1"])
    h, w = a.shape[:2]
    # Left projector: full grey at its left edge, fading out across the overlap on its right
    left = a[h // 2 - 40 : h // 2 + 40, int(w * 0.20)].astype(float).mean()
    right = a[h // 2 - 40 : h // 2 + 40, int(w * 0.98)].astype(float).mean()
    assert left > 100 and right < left * 0.6
    win.actions["test_pattern"].setChecked(False)


def test_arrow_keys_nudge_projector_in_pattern_mode(app, win):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    win.actions["test_pattern"].setChecked(True)
    pump(app, 0.5)
    pw = win.outputs.windows["proj2"]
    x0 = win.doc.scene.projector("proj2").transform.x
    QTest.keyClick(pw, Qt.Key.Key_Right)
    QTest.keyClick(pw, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    pump(app, 0.2)
    assert win.doc.scene.projector("proj2").transform.x == pytest.approx(x0 + 11)
    win.doc.undo.undo()                              # one merged step
    assert win.doc.scene.projector("proj2").transform.x == pytest.approx(x0)
    # Corner mode: Tab picks a corner, arrows move only that corner
    win.doc.update_item("projector", "proj2", lambda s: setattr(s.transform, "corners", s.transform.quad().tolist()),
                        "warp")
    pump(app, 0.1)
    QTest.keyClick(pw, Qt.Key.Key_Tab)              # TL
    QTest.keyClick(pw, Qt.Key.Key_Tab)              # TR
    before = [list(c) for c in win.doc.scene.projector("proj2").transform.corners]
    QTest.keyClick(pw, Qt.Key.Key_Down)
    after = win.doc.scene.projector("proj2").transform.corners
    assert after[1][1] == pytest.approx(before[1][1] + 1)
    assert after[0] == before[0] and after[2] == before[2]
    win.actions["test_pattern"].setChecked(False)


def test_blackout_fades_out_and_in(app, win):
    win.actions["outputs"].setChecked(True)
    pump(app, 0.5)
    outs = win.outputs
    win.actions["blackout"].setChecked(True)
    pump(app, 1.0)
    assert outs.level == 0.0
    assert grab(outs.windows["proj1"]).max() <= 2
    win.actions["blackout"].setChecked(False)
    pump(app, 1.0)
    assert outs.level == 1.0


def test_identify_and_scene_changes(app, win):
    win.actions["identify"].trigger()
    pump(app, 0.4)
    assert win.outputs.identify_until > time.perf_counter()
    # Disabling a projector closes its window; re-enabling brings it back
    win.doc.update_item("projector", "proj2", lambda s: setattr(s, "enabled", False), "off")
    pump(app, 0.3)
    assert "proj2" not in win.outputs.windows
    win.doc.undo.undo()
    pump(app, 0.5)
    assert win.outputs.windows["proj2"].isVisible()


def test_missing_display_falls_back_to_window(app, win):
    win.doc.update_item("projector", "proj1", lambda s: (setattr(s.output, "kind", "screen"),
                                                         setattr(s.output, "screen_name", "NO-SUCH-DISPLAY"),
                                                         setattr(s.output, "screen_index", 42)), "screen")
    win.actions["outputs"].setChecked(True)
    pump(app, 0.6)
    w1 = win.outputs.windows["proj1"]
    assert w1.state == "missing" and w1.isVisible() and "not connected" in w1.title()
