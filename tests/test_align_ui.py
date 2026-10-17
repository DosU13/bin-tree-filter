"""Align Perform window driven with simulated mouse / keyboard (opens a real window)."""

import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from tests.engine_cases import ROOT

pytestmark = pytest.mark.gui
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
    w.resize(1400, 850)
    w.show()
    pump(app, 0.8)
    w.view.fit()
    pump(app, 0.3)
    yield w
    w.doc.undo.setClean()       # no save prompt
    w.close()
    pump(app, 0.2)


def vp(win, x, y) -> QPoint:
    """Viewport point of world (x, y)."""
    return win.view.mapFromScene(QPointF(x, y))


def drag(app, win, start, end, mods=Qt.KeyboardModifier.NoModifier, steps=6):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    port = win.view.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, mods, start)
    for i in range(1, steps + 1):
        p = QPointF(start + (end - start) * (i / steps))
        # Move events carry the held button and modifiers, like a real drag
        ev = QMouseEvent(QEvent.Type.MouseMove, p, port.mapToGlobal(p), Qt.MouseButton.NoButton,
                         Qt.MouseButton.LeftButton, mods)
        QApplication.sendEvent(port, ev)
        pump(app, 0.01)
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, mods, end)
    pump(app, 0.1)


def test_window_runs_the_scene(app, win):
    pump(app, 1.0)
    r = win.session.last
    assert r is not None and set(r.camera_status.values()) == {"ok"}
    assert win.session.preview_image is not None
    assert win.windowTitle().startswith("example.json")


def test_drag_moves_camera_and_undo_restores(app, win):
    win.actions["snap"].setChecked(False)
    win.doc.select("camera", "cam2")
    pump(app, 0.1)
    cam = win.doc.scene.camera("cam2")
    x0 = cam.transform.x
    cx, cy = cam.transform.x + 300, cam.transform.y + 270
    drag(app, win, vp(win, cx, cy), vp(win, cx + 100, cy))
    moved = win.doc.scene.camera("cam2").transform.x
    assert moved == pytest.approx(x0 + 100, abs=4)
    assert win.doc.dirty
    win.doc.undo.undo()
    assert win.doc.scene.camera("cam2").transform.x == pytest.approx(x0)


def test_resize_handle_keeps_aspect_unless_shift(app, win):
    win.actions["snap"].setChecked(False)
    win.doc.select("projector", "proj1")
    pump(app, 0.1)
    t = win.doc.scene.projector("proj1").transform
    r = vp(win, t.x + t.width, t.y + t.height / 2)
    drag(app, win, r, r + QPoint(-60, 0))
    t2 = win.doc.scene.projector("proj1").transform
    assert t2.width < t.width
    assert t2.width / t2.height == pytest.approx(16 / 9, rel=1e-3)
    r = vp(win, t2.x + t2.width, t2.y + t2.height / 2)
    drag(app, win, r, r + QPoint(40, 0), mods=Qt.KeyboardModifier.ShiftModifier)
    t3 = win.doc.scene.projector("proj1").transform
    assert t3.height == pytest.approx(t2.height)
    assert t3.width > t2.width


def test_snapping_pulls_onto_stage_edge(app, win):
    win.actions["snap"].setChecked(True)
    win.doc.select("camera", "cam1")
    pump(app, 0.1)
    t = win.doc.scene.camera("cam1").transform
    start = vp(win, t.x + 100, t.y + 100)
    drag(app, win, start, start + QPoint(4, 0))      # a few px: snaps back to x = 0
    assert win.doc.scene.camera("cam1").transform.x == pytest.approx(0)


def test_param_hotkeys_on_canvas(app, win):
    win.view.setFocus()
    before = win.doc.scene.params["max_depth"]
    QTest.keyClick(win.view, "D")
    QTest.keyClick(win.view, "D")
    assert win.doc.scene.params["max_depth"] == min(12, before + 2)
    win.doc.undo.undo()                                # merged into one step
    assert win.doc.scene.params["max_depth"] == before
    QTest.keyClick(win.view, "g")
    assert win.doc.scene.params["cord"] is True


def test_inspector_edits_and_rotation_swap(app, win):
    win.doc.select("camera", "cam3")
    pump(app, 0.1)
    page = win.inspector.camera
    page.x.setValue(950)
    assert win.doc.scene.camera("cam3").transform.x == pytest.approx(950)
    w0, h0 = win.doc.scene.camera("cam3").transform.width, win.doc.scene.camera("cam3").transform.height
    page.rotation.setCurrentIndex(1)                   # 90°
    t = win.doc.scene.camera("cam3").transform
    assert t.rotation == 90 and (t.width, t.height) == pytest.approx((h0, w0))
    page.corner_mode.setChecked(True)
    assert win.doc.scene.camera("cam3").transform.corners is not None


def test_stage_fit_to_projectors(app, win):
    win.doc.select("stage", "")
    pump(app, 0.1)
    win.inspector.stage.fit_btn.click()
    st = win.doc.scene.stage
    assert st.rows == 1 and st.cell_size == pytest.approx(540) and st.cols == 3


def test_add_remove_duplicate_undo(app, win):
    from core.scene import CameraSpec, SourceSpec, Transform
    n = len(win.doc.scene.cameras)
    win.doc.add_item("camera", CameraSpec(id="", name="Extra", source=SourceSpec(kind="image", path="me.jpg"),
                                          transform=Transform(0, 0, 400, 300)))
    pump(app, 0.3)
    assert len(win.doc.scene.cameras) == n + 1
    new_id = win.doc.scene.cameras[-1].id
    assert ("camera", new_id) in win.canvas.items_by_key
    win.edit_duplicate()
    assert len(win.doc.scene.cameras) == n + 2
    win.edit_delete()
    win.doc.undo.undo()
    win.doc.undo.undo()
    win.doc.undo.undo()
    assert len(win.doc.scene.cameras) == n
    assert ("camera", new_id) not in win.canvas.items_by_key


def test_save_and_reopen(app, win, tmp_path):
    win.doc.set_param("max_depth", 7)
    path = str(tmp_path / "saved.json")
    assert win._save_to(path)
    assert not win.doc.dirty
    win.doc.new()
    assert win.open_path(path)
    assert win.doc.scene.params["max_depth"] == 7
    assert len(win.doc.scene.cameras) == 3


def test_add_camera_dialog_probes_a_file(app):
    from align_perform.dialogs.add_camera import AddCameraDialog
    dlg = AddCameraDialog(None)
    dlg.tabs.setCurrentIndex(1)
    dlg.path.setText(os.path.join(ROOT, "scenes", "media", "cam_mid.mp4"))
    dlg._test()
    deadline = time.time() + 10
    while time.time() < deadline and dlg._frame_size == (0, 0):
        pump(app, 0.05)
    assert dlg._frame_size == (480, 360)
    dlg._accept()
    spec = dlg.result_spec
    assert spec.kind == "video" and (spec.width, spec.height) == (480, 360) and spec.fps == 15


def test_add_projector_dialog_window_output(app):
    from align_perform.dialogs.add_projector import AddProjectorDialog
    dlg = AddProjectorDialog(None)
    dlg.list.setCurrentRow(0)              # "Window (testing)"
    dlg.w.setValue(1280)
    dlg.h.setValue(800)
    dlg._accept()
    out = dlg.result_output
    assert out.kind == "window" and (out.width, out.height) == (1280, 800)
    assert dlg.list.count() >= 2           # at least one display listed too


def test_look_panel_edits_params_and_follows_mode(app, win):
    panel = win.look_panel
    panel.widgets["threshold"].spin.setValue(0.5)
    assert win.doc.scene.params["threshold"] == pytest.approx(0.5)
    panel.widgets["cord"].setChecked(True)
    assert win.doc.scene.params["cord"] is True
    win.doc.set_param("mode", "depth")
    pump(app, 0.05)
    assert panel.widgets["colors"].isHidden() and not panel.widgets["depth_min"].isHidden()
    assert panel.groups["Colour"].isHidden()
    win.doc.set_param("mode", "colour")
    pump(app, 0.05)
    assert not panel.widgets["colors"].isHidden()
    # Undo returns the slider too
    win.doc.undo.undo()
    win.doc.undo.undo()
    win.doc.undo.undo()
    assert panel.widgets["cord"].isChecked() is False


def test_looks_save_crossfade_rename_delete(app, win):
    looks = win.looks_panel
    win.doc.set_param("max_depth", 5)
    win.doc.save_look("low")
    assert win.doc.scene.looks["low"]["max_depth"] == 5
    win.doc.set_param("max_depth", 11)
    looks.crossfade("low", 0.4)
    assert looks.fading
    pump(app, 0.2)
    assert win.doc.scene.params["max_depth"] == 11          # document untouched mid-fade
    pump(app, 0.5)
    assert not looks.fading and win.doc.scene.params["max_depth"] == 5
    win.doc.rename_look("low", "quiet")
    assert "quiet" in win.doc.scene.looks and "low" not in win.doc.scene.looks
    win.doc.delete_look("quiet")
    assert "quiet" not in win.doc.scene.looks
    win.doc.undo.undo()
    assert "quiet" in win.doc.scene.looks


def test_unknown_camera_size_is_learned_from_frames(app, win):
    from core.scene import CameraSpec, SourceSpec, Transform
    win.doc.add_item("camera", CameraSpec(id="", name="Untested", source=SourceSpec(
        kind="video", path=os.path.join(ROOT, "scenes", "media", "cam_left.mp4")),
        transform=Transform(0, 0, 640, 360)))
    cid = win.doc.scene.cameras[-1].id
    deadline = time.time() + 8
    while time.time() < deadline and win.doc.scene.camera(cid).source.width == 0:
        pump(app, 0.1)
    cam = win.doc.scene.camera(cid)
    assert (cam.source.width, cam.source.height) == (480, 360)
    assert cam.transform.width / cam.transform.height == pytest.approx(4 / 3)
