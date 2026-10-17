"""Align Perform main window: canvas in the middle, scene list left, inspector / look right."""

from __future__ import annotations

import logging
import os

import cv2
from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget,
                               QToolBar, QWidget)

from align_perform.actions import MENUS, TOOLBAR, build_actions
from align_perform.canvas.scene import CanvasScene
from align_perform.canvas.view import CanvasView
from align_perform.dialogs.add_camera import AddCameraDialog, _Probe
from align_perform.dialogs.add_projector import AddProjectorDialog
from align_perform.document import SceneDocument
from align_perform.panels.inspector import Inspector
from align_perform.panels.scene_tree import SceneTree
from align_perform.panels.status import StatusPanel
from align_perform.runtime.session import LiveSession
from core import params as P
from core.scene import CameraSpec, ProjectorSpec, SceneError, Transform

log = logging.getLogger(__name__)

SCENE_FILTER = "Scenes (*.json);;All files (*)"
VIEW_MODES = {"view_ornament": "ornament", "view_input": "input", "view_labels": "labels",
              "view_cameras": "cameras"}
MAX_RECENT = 8


class MainWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.settings = QSettings()
        self.doc = SceneDocument(self)
        self.session = LiveSession(self.doc.scene, self)
        self.canvas = CanvasScene(self.doc, self)
        self.view = CanvasView(self.canvas, lambda: self.session)
        self.setCentralWidget(self.view)
        self.outputs = None                     # ProjectorOutputs (phase 6)

        # Docks
        self.tree = SceneTree(self.doc)
        self.inspector = Inspector(self.doc)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.inspector, "Inspector")
        self._add_look_tabs()
        self.left_dock = self._dock("Scene", self.tree, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.right_dock = self._dock("Properties", self.right_tabs, Qt.DockWidgetArea.RightDockWidgetArea)

        # Status bar
        self.status_panel = StatusPanel(self.doc)
        self.statusBar().addPermanentWidget(self.status_panel)

        # Actions, menus, toolbar
        self.actions = build_actions(self)
        self._build_menus()
        self._build_toolbar()

        # Wiring
        self._scene_timer = QTimer(self)
        self._scene_timer.setSingleShot(True)
        self._scene_timer.setInterval(40)
        self._scene_timer.timeout.connect(self._push_scene)
        self.doc.sceneChanged.connect(self._scene_timer.start)
        self.doc.fileChanged.connect(self._update_title)
        self.doc.selectionChanged.connect(self._update_actions)
        self.session.frameReady.connect(self._on_frame)
        self.session.previewReady.connect(self.view.viewport().update)
        self.session.message.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self.view.viewChanged.connect(lambda r, w, h: self.session.set_view(r, (w, h)))
        self.view.message.connect(lambda m: self.statusBar().showMessage(m, 2000))
        self.tree.addCamera.connect(self.add_camera)
        self.tree.addProjector.connect(self.add_projector)
        self.tree.focusItem.connect(self._focus_item)
        self.inspector.camera.changeSource.connect(self.change_source)
        self.inspector.camera.probeSource.connect(self.probe_source)

        self.resize(1500, 900)
        self._restore_settings()
        self._update_title()
        self._update_actions()
        if path:
            self.open_path(path)
        QTimer.singleShot(0, self.view.fit)

    # -- construction --------------------------------------------------------

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(area, dock)
        return dock

    def _add_look_tabs(self) -> None:
        """Look and looks panels (all live.py options); see panels/look_panel.py."""
        try:
            from align_perform.panels.look_panel import LookPanel
            from align_perform.panels.looks_panel import LooksPanel
        except ImportError:
            return
        self.look_panel = LookPanel(self.doc)
        self.looks_panel = LooksPanel(self.doc, self)
        self.right_tabs.addTab(self.look_panel, "Look")
        self.right_tabs.addTab(self.looks_panel, "Looks")

    def _build_menus(self) -> None:
        bar = self.menuBar()
        for title, names in MENUS:
            menu = bar.addMenu(title)
            for name in names:
                if name == "-":
                    menu.addSeparator()
                elif name == "recent":
                    self.recent_menu = menu.addMenu("Open &recent")
                    self._fill_recent()
                elif name == "panels":
                    menu.addAction(self.left_dock.toggleViewAction())
                    menu.addAction(self.right_dock.toggleViewAction())
                else:
                    menu.addAction(self.actions[name])

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setObjectName("main-toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        for name in TOOLBAR:
            if name == "-":
                tb.addSeparator()
            elif name == "view_mode_combo":
                tb.addWidget(QLabel(" View: "))
                self.mode_combo = QComboBox()
                for action_name, mode in VIEW_MODES.items():
                    self.mode_combo.addItem(self.actions[action_name].text().replace("&", ""), action_name)
                self.mode_combo.currentIndexChanged.connect(
                    lambda _: self.actions[self.mode_combo.currentData()].setChecked(True))
                tb.addWidget(self.mode_combo)
            else:
                tb.addAction(self.actions[name])

    # -- settings ------------------------------------------------------------

    def _restore_settings(self) -> None:
        geom = self.settings.value("geometry")
        state = self.settings.value("state")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)
        if isinstance(state, QByteArray):
            self.restoreState(state)

    def _recent(self) -> list[str]:
        value = self.settings.value("recent", [])
        if isinstance(value, str):
            value = [value]
        return [p for p in (value or []) if os.path.exists(p)]

    def _add_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        recent = [p for p in self._recent() if os.path.normcase(p) != os.path.normcase(path)]
        self.settings.setValue("recent", [path] + recent[: MAX_RECENT - 1])
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = self._recent()
        for p in recent:
            self.recent_menu.addAction(p, lambda p=p: self.open_path(p) if self._confirm_discard() else None)
        self.recent_menu.setEnabled(bool(recent))

    # -- state ---------------------------------------------------------------

    def _update_title(self) -> None:
        self.setWindowTitle(f"{self.doc.title()} — Align Perform")

    def _update_actions(self) -> None:
        sel = self.doc.selection
        item = sel is not None and sel[0] in ("camera", "projector")
        self.actions["duplicate"].setEnabled(item)
        self.actions["delete"].setEnabled(item)

    def _push_scene(self) -> None:
        self.session.set_scene(self.doc.scene)
        if self.outputs is not None:
            self.outputs.set_scene(self.doc.scene)

    def _on_frame(self, r) -> None:
        self.status_panel.update_frame(r, self.session.paused)
        self.canvas.set_camera_status(r.camera_status)
        self.canvas.set_thumbnails(r.thumbnails)
        self.inspector.camera.set_statuses(r.camera_status)
        if self.outputs is not None:
            self.outputs.push(r)

    def _focus_item(self, kind: str, item_id: str) -> None:
        if kind == "stage":
            self.view.fit(self.canvas.stage_item.rect())
            return
        item = self.canvas.items_by_key.get((kind, item_id))
        if item is not None:
            self.view.fit(item.boundingRect())

    # -- file ----------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if not self.doc.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", f"Save changes to {self.doc.title().rstrip(' *')}?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.file_save()
        return r == QMessageBox.StandardButton.Discard

    def file_new(self) -> None:
        if self._confirm_discard():
            self.doc.new()
            QTimer.singleShot(0, self.view.fit)

    def file_open(self) -> None:
        if not self._confirm_discard():
            return
        start = os.path.dirname(self.doc.path) if self.doc.path else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", start, SCENE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        restore = False
        if self.doc.has_newer_autosave(path):
            r = QMessageBox.question(self, "Recover changes",
                                     "There are newer unsaved changes from a previous session. Restore them?")
            restore = r == QMessageBox.StandardButton.Yes
        try:
            self.doc.load(path, from_autosave=restore)
        except (OSError, SceneError) as exc:
            QMessageBox.warning(self, "Cannot open scene", str(exc))
            return False
        self._add_recent(path)
        self.statusBar().showMessage(f"Opened {path}", 4000)
        QTimer.singleShot(0, self.view.fit)
        return True

    def file_save(self) -> bool:
        if not self.doc.path:
            return self.file_save_as()
        return self._save_to(self.doc.path)

    def file_save_as(self) -> bool:
        start = self.doc.path or os.path.join(os.getcwd(), "scenes", "untitled.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", start, SCENE_FILTER)
        if not path:
            return False
        if not path.lower().endswith(".json"):
            path += ".json"
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        try:
            self.doc.save(path)
        except (OSError, SceneError) as exc:
            QMessageBox.warning(self, "Cannot save scene", str(exc))
            return False
        self._add_recent(path)
        self.statusBar().showMessage(f"Saved {path}", 4000)
        return True

    def file_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export snapshot to folder",
                                                  os.path.dirname(self.doc.path or os.getcwd()))
        if not folder:
            return
        written = []
        for p in self.doc.scene.enabled_projectors():
            img = self.session.render_projector(p)
            if img is not None:
                f = os.path.join(folder, f"{p.id}.png")
                cv2.imwrite(f, img)
                written.append(f)
        img = self.session.render_stage(min(4096, int(self.doc.scene.stage.width)))
        if img is not None:
            f = os.path.join(folder, "stage.png")
            cv2.imwrite(f, img)
            written.append(f)
        self.statusBar().showMessage(f"Wrote {len(written)} images to {folder}", 5000)

    # -- edit ----------------------------------------------------------------

    def edit_duplicate(self) -> None:
        sel = self.doc.selection
        if sel and sel[0] in ("camera", "projector"):
            self.doc.duplicate_item(*sel)

    def edit_delete(self) -> None:
        sel = self.doc.selection
        if sel and sel[0] in ("camera", "projector"):
            self.doc.remove_item(*sel)

    def edit_select_stage(self) -> None:
        self.doc.select("stage", "")

    # -- add -----------------------------------------------------------------

    def add_camera(self) -> None:
        n = len(self.doc.scene.cameras)
        dlg = AddCameraDialog(self, self.doc.scene.resolve, name=f"Camera {n + 1}")
        if not dlg.exec() or dlg.result_spec is None:
            return
        cam = CameraSpec(id=self.doc.scene.new_id("cam"), name=dlg.name.text().strip() or f"Camera {n + 1}",
                         source=dlg.result_spec)
        st = self.doc.scene.stage
        aspect = cam.display_aspect()
        h = st.height
        w = h * aspect
        if w > st.width:
            w, h = st.width, st.width / aspect
        # Fan new cameras out across the stage, overlapping by a quarter
        x = st.x + min(n * w * 0.75, max(0.0, st.width - w))
        cam.transform = Transform(x=x, y=st.y, width=w, height=h)
        cam.feather = round(w * 0.12)
        self.doc.add_item("camera", cam)

    def add_projector(self) -> None:
        n = len(self.doc.scene.projectors)
        dlg = AddProjectorDialog(self, name=f"Projector {n + 1}")
        if not dlg.exec() or dlg.result_output is None:
            return
        out = dlg.result_output
        proj = ProjectorSpec(id=self.doc.scene.new_id("proj"),
                             name=dlg.name.text().strip() or f"Projector {n + 1}", output=out)
        st = self.doc.scene.stage
        aspect = out.width / out.height
        h = st.height
        w = h * aspect
        x = st.x + n * w * 0.85
        proj.transform = Transform(x=x, y=st.y, width=w, height=h)
        self.doc.add_item("projector", proj)

    def change_source(self, cam_id: str) -> None:
        cam = self.doc.scene.camera(cam_id)
        dlg = AddCameraDialog(self, self.doc.scene.resolve, spec=cam.source, name=cam.name)
        if not dlg.exec() or dlg.result_spec is None:
            return
        name = dlg.name.text().strip() or cam.name
        new_src = dlg.result_spec

        def mutate(s):
            old_aspect = s.display_aspect()
            s.source = new_src
            s.name = name
            t = s.transform
            if t.corners is None and abs(s.display_aspect() - old_aspect) > 1e-3:
                t.height = t.width / s.display_aspect()
        self.doc.update_item("camera", cam_id, mutate, "Change camera source")

    def probe_source(self, cam_id: str) -> None:
        cam = self.doc.scene.camera(cam_id)
        probe = _Probe(self)

        def done(frame, fps, error, cam_id=cam_id):
            probe.deleteLater()
            if frame is None:
                self.statusBar().showMessage(f"{cam.name or cam_id}: {error}", 6000)
                return
            h, w = frame.shape[:2]

            def mutate(s):
                s.source.width, s.source.height = w, h
                if fps and s.source.kind != "webcam":
                    s.source.fps = round(fps, 2)
            self.doc.update_item("camera", cam_id, mutate, "Detect camera size")
            self.statusBar().showMessage(f"{cam.name or cam_id}: {w}×{h}", 4000)
        probe.done.connect(done)
        self.statusBar().showMessage(f"Opening {cam.source.label()}…")
        probe.start(cam.source, self.doc.scene.resolve)

    # -- view ----------------------------------------------------------------

    def view_mode(self, name: str, on: bool) -> None:
        if not on:
            return
        mode = VIEW_MODES[name]
        self.canvas.show_camera_images = mode == "cameras"
        self.session.set_preview_mode("ornament" if mode == "cameras" else mode)
        if mode != "cameras":
            for item in self.canvas.items_by_key.values():
                if hasattr(item, "set_frame"):
                    item.set_frame(None)
        if hasattr(self, "mode_combo"):
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(name))
            self.mode_combo.blockSignals(False)
        self.view.viewport().update()

    def view_toggles(self, name: str, on: bool) -> None:
        if name == "show_cameras":
            self.canvas.set_visibility(cameras=on)
        elif name == "show_projectors":
            self.canvas.set_visibility(projectors=on)
        elif name == "show_grid":
            self.canvas.show_grid = on
            self.canvas.update()
        elif name == "snap":
            self.canvas.snap = on

    def view_fit(self) -> None:
        self.view.fit()

    def view_zoom_in(self) -> None:
        self.view.zoom(1.25)

    def view_zoom_out(self) -> None:
        self.view.zoom(1 / 1.25)

    # -- output (projector windows, phase 6) ----------------------------------

    def _ensure_outputs(self):
        if self.outputs is None:
            try:
                from align_perform.runtime.outputs import ProjectorOutputs
            except ImportError:
                self.statusBar().showMessage("Projector outputs are not available", 4000)
                return None
            self.outputs = ProjectorOutputs(self.doc.scene, self.session, key_handler=self.view.keyPressEvent)
            self.outputs.statesChanged.connect(self.status_panel.set_projectors)
            self.outputs.closedByUser.connect(lambda: self.actions["outputs"].setChecked(False))
            self.outputs.adjustRequested.connect(self._adjust_projector)
            if self.session.last is not None:
                self.outputs.push(self.session.last)
        return self.outputs

    def _adjust_projector(self, proj_id: str, corner, dx: float, dy: float) -> None:
        """Arrow keys on a projector showing the test pattern: move it, or one warp corner."""
        def mutate(spec):
            t = spec.transform
            if t.corners is not None and corner is not None:
                t.corners[corner][0] += dx
                t.corners[corner][1] += dy
            else:
                t.x += dx
                t.y += dy
                if t.corners is not None:
                    t.corners = [[x + dx, y + dy] for x, y in t.corners]
        self.doc.update_item("projector", proj_id, mutate, "Nudge projector", merge_key=f"projector-nudge-{proj_id}")

    def output_toggle(self, name: str, on: bool) -> None:
        outputs = self._ensure_outputs()
        if outputs is not None:
            outputs.set_visible(on)

    def output_test_pattern(self, name: str, on: bool) -> None:
        outputs = self._ensure_outputs()
        if outputs is not None:
            outputs.set_test_pattern(on)
            if on and not self.actions["outputs"].isChecked():
                self.actions["outputs"].setChecked(True)

    def output_identify(self) -> None:
        outputs = self._ensure_outputs()
        if outputs is not None:
            if not self.actions["outputs"].isChecked():
                self.actions["outputs"].setChecked(True)
            outputs.identify()

    def output_blackout(self, name: str, on: bool) -> None:
        outputs = self._ensure_outputs()
        if outputs is not None:
            outputs.set_blackout(on)

    # -- run / help ----------------------------------------------------------

    def run_pause(self, name: str, on: bool) -> None:
        self.session.set_paused(on)

    def help_shortcuts(self) -> None:
        rows = []
        for mode in ("colour", "depth"):
            for p in P.PARAMS:
                if p.keys and p.mode in (None, mode):
                    keys = " / ".join(p.keys)
                    row = f"<tr><td><b>{keys}</b></td><td>{p.label}</td><td>{p.mode or ''}</td></tr>"
                    if row not in rows:
                        rows.append(row)
        canvas = ("<tr><td><b>wheel</b></td><td>zoom at pointer</td></tr>"
                  "<tr><td><b>middle-drag / Space+drag</b></td><td>pan</td></tr>"
                  "<tr><td><b>F</b></td><td>fit stage</td></tr>"
                  "<tr><td><b>arrows</b></td><td>nudge selection 1 px (Shift: 10)</td></tr>"
                  "<tr><td><b>Shift while resizing</b></td><td>free aspect</td></tr>"
                  "<tr><td><b>Alt while resizing</b></td><td>from centre</td></tr>"
                  "<tr><td><b>Ctrl while dragging</b></td><td>no snapping</td></tr>"
                  "<tr><td><b>Alt+click</b></td><td>select the item underneath</td></tr>")
        QMessageBox.information(self, "Keyboard shortcuts",
                                "<p>With the canvas focused, the live.py keys change the look:</p>"
                                f"<table cellpadding=3>{''.join(rows)}</table>"
                                f"<p>Canvas:</p><table cellpadding=3>{canvas}</table>")

    def help_about(self) -> None:
        QMessageBox.about(self, "Align Perform",
                          "<b>Align Perform</b><br>Set up cameras and projectors for the Kyrgyz ornament "
                          "filter, then play the scene with <code>perform.py</code>.")

    # -- close ---------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        if self.outputs is not None:
            self.outputs.close()
        self.session.close()
        event.accept()
