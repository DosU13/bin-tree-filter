"""Inspector: properties of the selected stage, camera or projector.

Every edit is an undoable document change; repeated edits of one field merge
into a single undo step.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from align_perform.panels.widgets import HintLabel, combo, dspin, ispin, set_quiet
from align_perform.runtime.screens import list_screens
from core.scene import ROTATIONS

STATUS_TEXT = {"ok": "running", "starting": "starting…", "stalled": "stalled (no new frames)",
               "reconnecting": "reconnecting…", "stopped": "stopped", "disabled": "disabled"}


class Page(QWidget):
    kind = ""

    def __init__(self, doc) -> None:
        super().__init__()
        self.doc = doc
        self.item_id: str | None = None
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(6, 6, 6, 6)

    def group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.outer.addWidget(box)
        return form

    def spec(self):
        return self.doc.scene.item(self.kind, self.item_id)

    def edit(self, label: str, mutate, merge: bool = True) -> None:
        if self.item_id is None:
            return
        key = f"inspector-{self.kind}-{self.item_id}-{label}" if merge else None
        self.doc.update_item(self.kind, self.item_id, mutate, label, merge_key=key)

    def show_item(self, item_id: str | None) -> None:
        self.item_id = item_id
        self.refresh()

    def refresh(self) -> None:
        raise NotImplementedError


class PlacementMixin:
    """Position / size / 4-corner warp fields shared by cameras and projectors."""

    def build_placement(self) -> None:
        form = self.group("Placement (world px)")
        self.x, self.y = dspin(-1e6, 1e6, 1, 1), dspin(-1e6, 1e6, 1, 1)
        self.w, self.h = dspin(1, 1e6, 1, 1), dspin(1, 1e6, 1, 1)
        row = QHBoxLayout()
        row.addWidget(self.x)
        row.addWidget(self.y)
        form.addRow("X, Y", row)
        row = QHBoxLayout()
        row.addWidget(self.w)
        row.addWidget(self.h)
        form.addRow("Width, height", row)
        self.keep_aspect = QCheckBox("Keep aspect ratio")
        self.keep_aspect.setChecked(True)
        form.addRow("", self.keep_aspect)
        self.corner_mode = QCheckBox("4-corner warp")
        self.corner_mode.setToolTip("Drag each corner separately (keystone / angled surfaces)")
        form.addRow("", self.corner_mode)
        self.corner_box = QWidget()
        cform = QFormLayout(self.corner_box)
        cform.setContentsMargins(0, 0, 0, 0)
        self.corner_spins = []
        for name in ("Top left", "Top right", "Bottom right", "Bottom left"):
            cx, cy = dspin(-1e6, 1e6, 1, 1), dspin(-1e6, 1e6, 1, 1)
            row = QHBoxLayout()
            row.addWidget(cx)
            row.addWidget(cy)
            cform.addRow(name, row)
            self.corner_spins.append((cx, cy))
        form.addRow(self.corner_box)
        self.reset_aspect = QPushButton("Match aspect ratio")
        form.addRow("", self.reset_aspect)

        self.x.valueChanged.connect(lambda v: self.edit("X", lambda s: _move(s.transform, dx=v - s.transform.bounds()[0])))
        self.y.valueChanged.connect(lambda v: self.edit("Y", lambda s: _move(s.transform, dy=v - s.transform.bounds()[1])))
        self.w.valueChanged.connect(lambda v: self.edit("Width", lambda s: self._resize(s, w=v)))
        self.h.valueChanged.connect(lambda v: self.edit("Height", lambda s: self._resize(s, h=v)))
        self.corner_mode.toggled.connect(self._toggle_corners)
        for i, (cx, cy) in enumerate(self.corner_spins):
            cx.valueChanged.connect(lambda v, i=i: self.edit("Corner", lambda s: _set_corner(s.transform, i, x=v)))
            cy.valueChanged.connect(lambda v, i=i: self.edit("Corner", lambda s: _set_corner(s.transform, i, y=v)))
        self.reset_aspect.clicked.connect(lambda: self.edit("Match aspect", lambda s: self._resize(s, w=s.transform.width, force=True), merge=False))

    def aspect(self, spec) -> float:
        raise NotImplementedError

    def _resize(self, spec, w: float | None = None, h: float | None = None, force: bool = False) -> None:
        t = spec.transform
        if t.corners is not None:
            return
        a = self.aspect(spec)
        keep = self.keep_aspect.isChecked() or force
        if w is not None:
            t.width = w
            if keep:
                t.height = w / a
        elif h is not None:
            t.height = h
            if keep:
                t.width = h * a

    def _toggle_corners(self, on: bool) -> None:
        def mutate(s):
            t = s.transform
            if on and t.corners is None:
                t.corners = t.quad().tolist()
            elif not on and t.corners is not None:
                x, y, w, h = t.bounds()
                t.set_rect(x, y, w, h)
        self.edit("4-corner warp", mutate, merge=False)

    def refresh_placement(self, spec) -> None:
        t = spec.transform
        x, y, w, h = t.bounds()
        for widget, value in ((self.x, x), (self.y, y), (self.w, w), (self.h, h)):
            set_quiet(widget, value)
        corners = t.corners is not None
        set_quiet(self.corner_mode, corners)
        self.corner_box.setVisible(corners)
        self.w.setEnabled(not corners)
        self.h.setEnabled(not corners)
        self.reset_aspect.setEnabled(not corners)
        if corners:
            for (cx, cy), (px, py) in zip(self.corner_spins, t.corners):
                set_quiet(cx, px)
                set_quiet(cy, py)


def _move(t, dx: float = 0.0, dy: float = 0.0) -> None:
    t.x += dx
    t.y += dy
    if t.corners is not None:
        t.corners = [[x + dx, y + dy] for x, y in t.corners]


def _set_corner(t, i: int, x: float | None = None, y: float | None = None) -> None:
    if t.corners is None:
        return
    if x is not None:
        t.corners[i][0] = x
    if y is not None:
        t.corners[i][1] = y


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

class EmptyPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(HintLabel("Select the stage, a camera or a projector on the canvas or in the scene list.\n\n"
                                "Add cameras and projectors from the Add menu."))
        lay.addStretch(1)


class StagePage(Page):
    kind = "stage"

    def __init__(self, doc) -> None:
        super().__init__(doc)
        form = self.group("Stage")
        self.x, self.y = dspin(-1e6, 1e6, 1, 1), dspin(-1e6, 1e6, 1, 1)
        self.cols, self.rows = ispin(1, 64), ispin(1, 64)
        self.cell = dspin(16, 20000, 10, 1, " px")
        self.levels = ispin(4, 14)
        self.levels.setToolTip("Deepest triangle level; each level doubles the tree's work")
        row = QHBoxLayout()
        row.addWidget(self.x)
        row.addWidget(self.y)
        form.addRow("X, Y", row)
        row = QHBoxLayout()
        row.addWidget(self.cols)
        row.addWidget(self.rows)
        form.addRow("Cells across, down", row)
        form.addRow("Cell size", self.cell)
        form.addRow("Levels", self.levels)
        self.size_label = QLabel()
        form.addRow("Size", self.size_label)
        self.fit_btn = QPushButton("Fit stage to projectors")
        form.addRow("", self.fit_btn)
        self.outer.addWidget(HintLabel("The ornament exists only on the stage. Cell size sets the ornament's "
                                       "scale; cells are the bintree's root squares."))
        self.outer.addStretch(1)
        for w, name in ((self.x, "x"), (self.y, "y"), (self.cols, "cols"), (self.rows, "rows"),
                        (self.cell, "cell_size"), (self.levels, "levels")):
            w.valueChanged.connect(lambda v, name=name: self._set(name, v))
        self.fit_btn.clicked.connect(self._fit)

    def _set(self, name: str, value) -> None:
        import copy
        st = copy.deepcopy(self.doc.scene.stage)
        setattr(st, name, value)
        self.doc.set_stage(st, f"Stage {name}", merge_key=f"stage-{name}")

    def _fit(self) -> None:
        import copy
        scene = self.doc.scene
        items = scene.projectors or scene.cameras
        if not items:
            return
        quads = [i.transform.quad() for i in items]
        x0 = min(q[:, 0].min() for q in quads)
        y0 = min(q[:, 1].min() for q in quads)
        x1 = max(q[:, 0].max() for q in quads)
        y1 = max(q[:, 1].max() for q in quads)
        st = copy.deepcopy(scene.stage)
        # One row of cells as tall as the projectors, as many columns as needed
        st.cell_size = float(y1 - y0)
        st.rows = 1
        st.cols = max(1, math.ceil((x1 - x0) / st.cell_size - 1e-6))
        st.x, st.y = float(x0), float(y0)
        self.doc.set_stage(st, "Fit stage to projectors")

    def refresh(self) -> None:
        st = self.doc.scene.stage
        for w, v in ((self.x, st.x), (self.y, st.y), (self.cols, st.cols), (self.rows, st.rows),
                     (self.cell, st.cell_size), (self.levels, st.levels)):
            set_quiet(w, v)
        self.size_label.setText(f"{st.width:g} × {st.height:g} px")


class CameraPage(Page, PlacementMixin):
    kind = "camera"
    changeSource = Signal(str)       # camera id
    probeSource = Signal(str)

    def __init__(self, doc) -> None:
        super().__init__(doc)
        form = self.group("Camera")
        self.name = QLineEdit()
        self.enabled = QCheckBox("Enabled")
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.resolution = QLabel()
        self.status = QLabel()
        btns = QHBoxLayout()
        self.change_btn = QPushButton("Change source…")
        self.probe_btn = QPushButton("Re-detect size")
        btns.addWidget(self.change_btn)
        btns.addWidget(self.probe_btn)
        form.addRow("Name", self.name)
        form.addRow("", self.enabled)
        form.addRow("Source", self.source)
        form.addRow("Frame size", self.resolution)
        form.addRow("Status", self.status)
        form.addRow(btns)

        self.build_placement()
        form2 = self.group("Image")
        self.rotation = combo([f"{r}°" for r in ROTATIONS], list(ROTATIONS))
        self.mirror = QCheckBox("Mirror horizontally")
        self.brightness = ispin(-100, 100, 5)
        self.contrast = dspin(0.1, 3.0, 0.05, 2)
        self.saturation = dspin(0.0, 3.0, 0.05, 2)
        self.feather = dspin(0, 5000, 10, 0, " px")
        self.feather.setToolTip("Width over which this camera fades into overlapping cameras")
        form2.addRow("Rotation", self.rotation)
        form2.addRow("", self.mirror)
        form2.addRow("Brightness", self.brightness)
        form2.addRow("Contrast", self.contrast)
        form2.addRow("Saturation", self.saturation)
        form2.addRow("Edge feather", self.feather)
        self.outer.addStretch(1)

        self.name.editingFinished.connect(lambda: self.edit("Rename", lambda s: setattr(s, "name", self.name.text()), merge=False))
        self.enabled.toggled.connect(lambda v: self.edit("Enable", lambda s: setattr(s, "enabled", v), merge=False))
        self.change_btn.clicked.connect(lambda: self.item_id and self.changeSource.emit(self.item_id))
        self.probe_btn.clicked.connect(lambda: self.item_id and self.probeSource.emit(self.item_id))
        self.rotation.currentIndexChanged.connect(lambda _: self.edit("Rotate", self._rotate, merge=False))
        self.mirror.toggled.connect(lambda v: self.edit("Mirror", lambda s: setattr(s.transform, "mirror", v), merge=False))
        self.brightness.valueChanged.connect(lambda v: self.edit("Brightness", lambda s: setattr(s.color, "brightness", v)))
        self.contrast.valueChanged.connect(lambda v: self.edit("Contrast", lambda s: setattr(s.color, "contrast", v)))
        self.saturation.valueChanged.connect(lambda v: self.edit("Saturation", lambda s: setattr(s.color, "saturation", v)))
        self.feather.valueChanged.connect(lambda v: self.edit("Feather", lambda s: setattr(s, "feather", v)))
        self._statuses: dict[str, str] = {}

    def aspect(self, spec) -> float:
        return spec.display_aspect()

    def _rotate(self, spec) -> None:
        new = self.rotation.currentData()
        t = spec.transform
        if (new - t.rotation) % 180 == 90 and t.corners is None:
            cx, cy = t.x + t.width / 2, t.y + t.height / 2
            t.width, t.height = t.height, t.width
            t.x, t.y = cx - t.width / 2, cy - t.height / 2
        t.rotation = new

    def set_statuses(self, statuses: dict[str, str]) -> None:
        self._statuses = statuses
        if self.item_id:
            st = "disabled" if not self.spec().enabled else statuses.get(self.item_id, "starting")
            self.status.setText(STATUS_TEXT.get(st, st))

    def refresh(self) -> None:
        if self.item_id is None:
            return
        try:
            s = self.spec()
        except KeyError:
            return
        set_quiet(self.name, s.name)
        set_quiet(self.enabled, s.enabled)
        self.source.setText(s.source.label())
        src = s.source
        self.resolution.setText(f"{src.width}×{src.height}" + (f" @ {src.fps:g} fps" if src.fps else "")
                                if src.width else "unknown (not detected yet)")
        self.set_statuses(self._statuses)
        self.refresh_placement(s)
        set_quiet(self.rotation, s.transform.rotation)
        set_quiet(self.mirror, s.transform.mirror)
        set_quiet(self.brightness, s.color.brightness)
        set_quiet(self.contrast, s.color.contrast)
        set_quiet(self.saturation, s.color.saturation)
        set_quiet(self.feather, s.feather)


class ProjectorPage(Page, PlacementMixin):
    kind = "projector"

    def __init__(self, doc) -> None:
        super().__init__(doc)
        form = self.group("Projector")
        self.name = QLineEdit()
        self.enabled = QCheckBox("Enabled")
        self.output_kind = combo(["Display (full screen)", "Window (testing)", "Off"], ["screen", "window", "none"])
        self.screen = combo([])
        self.refresh_screens_btn = QPushButton("↻")
        self.refresh_screens_btn.setFixedWidth(28)
        self.refresh_screens_btn.setToolTip("Look for displays again")
        row = QHBoxLayout()
        row.addWidget(self.screen, 1)
        row.addWidget(self.refresh_screens_btn)
        self.res_w, self.res_h = ispin(16, 16384), ispin(16, 16384)
        res = QHBoxLayout()
        res.addWidget(self.res_w)
        res.addWidget(self.res_h)
        form.addRow("Name", self.name)
        form.addRow("", self.enabled)
        form.addRow("Output", self.output_kind)
        form.addRow("Display", row)
        form.addRow("Resolution", res)

        self.build_placement()
        form2 = self.group("Blending")
        self.gain = dspin(0.0, 2.0, 0.01, 2)
        self.auto_feather = QCheckBox("Automatic (overlap width)")
        self.feather = dspin(0, 1e5, 10, 0, " px")
        self.gamma = dspin(1.0, 3.0, 0.05, 2)
        form2.addRow("Brightness gain", self.gain)
        form2.addRow("Edge blend", self.auto_feather)
        form2.addRow("Blend width", self.feather)
        form2.addRow("Gamma", self.gamma)
        self.outer.addStretch(1)

        self.name.editingFinished.connect(lambda: self.edit("Rename", lambda s: setattr(s, "name", self.name.text()), merge=False))
        self.enabled.toggled.connect(lambda v: self.edit("Enable", lambda s: setattr(s, "enabled", v), merge=False))
        self.output_kind.currentIndexChanged.connect(lambda _: self.edit("Output", self._set_output_kind, merge=False))
        self.screen.currentIndexChanged.connect(lambda _: self.edit("Display", self._set_screen, merge=False))
        self.refresh_screens_btn.clicked.connect(self.refresh)
        self.res_w.valueChanged.connect(lambda v: self.edit("Resolution", lambda s: self._set_res(s, w=v)))
        self.res_h.valueChanged.connect(lambda v: self.edit("Resolution", lambda s: self._set_res(s, h=v)))
        self.gain.valueChanged.connect(lambda v: self.edit("Gain", lambda s: setattr(s, "gain", v)))
        self.auto_feather.toggled.connect(lambda v: self.edit("Edge blend", lambda s: setattr(
            s.blend, "feather", None if v else self.feather.value()), merge=False))
        self.feather.valueChanged.connect(lambda v: self.edit("Blend width", lambda s: setattr(s.blend, "feather", v)))
        self.gamma.valueChanged.connect(lambda v: self.edit("Gamma", lambda s: setattr(s.blend, "gamma", v)))

    def aspect(self, spec) -> float:
        return spec.aspect()

    def _keep_height(self, spec) -> None:
        t = spec.transform
        if t.corners is None and self.keep_aspect.isChecked():
            t.height = t.width / spec.aspect()

    def _set_output_kind(self, spec) -> None:
        spec.output.kind = self.output_kind.currentData()
        if spec.output.kind == "screen":
            self._set_screen(spec)

    def _set_screen(self, spec) -> None:
        info = self.screen.currentData()
        if info is None or spec.output.kind != "screen":
            return
        o = spec.output
        o.screen_name, o.screen_serial, o.screen_index = info.name, info.serial, info.index
        o.width, o.height = info.width, info.height
        self._keep_height(spec)

    def _set_res(self, spec, w: int | None = None, h: int | None = None) -> None:
        if w is not None:
            spec.output.width = w
        if h is not None:
            spec.output.height = h
        self._keep_height(spec)

    def refresh(self) -> None:
        if self.item_id is None:
            return
        try:
            s = self.spec()
        except KeyError:
            return
        set_quiet(self.name, s.name)
        set_quiet(self.enabled, s.enabled)
        set_quiet(self.output_kind, s.output.kind)
        self.screen.blockSignals(True)
        self.screen.clear()
        found = False
        for info in list_screens():
            self.screen.addItem(info.label(), info)
            if (s.output.screen_serial and info.serial == s.output.screen_serial) or info.name == s.output.screen_name:
                self.screen.setCurrentIndex(self.screen.count() - 1)
                found = True
        if s.output.kind == "screen" and not found:
            self.screen.addItem(f"Not connected: {s.output.screen_name or s.output.screen_index}", None)
            self.screen.setCurrentIndex(self.screen.count() - 1)
        self.screen.blockSignals(False)
        self.screen.setEnabled(s.output.kind == "screen")
        set_quiet(self.res_w, s.output.width)
        set_quiet(self.res_h, s.output.height)
        self.res_w.setEnabled(s.output.kind != "screen")
        self.res_h.setEnabled(s.output.kind != "screen")
        self.refresh_placement(s)
        set_quiet(self.gain, s.gain)
        set_quiet(self.auto_feather, s.blend.feather is None)
        set_quiet(self.feather, s.blend.feather or 0)
        self.feather.setEnabled(s.blend.feather is not None)
        set_quiet(self.gamma, s.blend.gamma)


class Inspector(QScrollArea):
    """Shows the page for the document's selection."""

    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setWidgetResizable(True)
        self.stack = QStackedWidget()
        self.empty = EmptyPage()
        self.stage = StagePage(doc)
        self.camera = CameraPage(doc)
        self.projector = ProjectorPage(doc)
        for page in (self.empty, self.stage, self.camera, self.projector):
            self.stack.addWidget(page)
        self.setWidget(self.stack)
        doc.selectionChanged.connect(self._on_selection)
        doc.itemChanged.connect(self._on_item_changed)
        doc.stageChanged.connect(lambda: self.stack.currentWidget() is self.stage and self.stage.refresh())
        doc.itemsChanged.connect(self._on_selection)
        self._on_selection()

    def _on_selection(self) -> None:
        sel = self.doc.selection
        if sel is None:
            self.stack.setCurrentWidget(self.empty)
        elif sel[0] == "stage":
            self.stack.setCurrentWidget(self.stage)
            self.stage.refresh()
        else:
            page = self.camera if sel[0] == "camera" else self.projector
            self.stack.setCurrentWidget(page)
            page.show_item(sel[1])

    def _on_item_changed(self, kind: str, item_id: str) -> None:
        if self.doc.selection == (kind, item_id):
            (self.camera if kind == "camera" else self.projector).refresh()
