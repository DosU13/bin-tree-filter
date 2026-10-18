"""Every menu / toolbar action, declared as data: text, shortcut, handler, check state.

MainWindow implements the handlers (methods named in `handler`); build_actions
creates the QActions and wires them up.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QStyle


@dataclass(frozen=True)
class ActionDef:
    name: str
    text: str
    shortcut: str = ""
    handler: str = ""
    checkable: bool = False
    checked: bool = False
    group: str = ""                 # exclusive group (radio items)
    icon: str = ""                  # QStyle.StandardPixmap name
    tip: str = ""


ACTIONS: tuple[ActionDef, ...] = (
    # File
    ActionDef("new", "&New scene", "Ctrl+N", "file_new", icon="SP_FileIcon"),
    ActionDef("open", "&Open…", "Ctrl+O", "file_open", icon="SP_DialogOpenButton"),
    ActionDef("save", "&Save", "Ctrl+S", "file_save", icon="SP_DialogSaveButton"),
    ActionDef("save_as", "Save &as…", "Ctrl+Shift+S", "file_save_as"),
    ActionDef("export", "&Export snapshot…", "Ctrl+E", "file_export",
              tip="Write each projector's current image (and the whole stage) as PNG files"),
    ActionDef("quit", "&Quit", "Ctrl+Q", "close"),
    # Edit
    ActionDef("duplicate", "&Duplicate", "Ctrl+D", "edit_duplicate"),
    ActionDef("delete", "De&lete", "Delete", "edit_delete"),
    ActionDef("select_stage", "Select &stage", "Ctrl+Shift+A", "edit_select_stage"),
    # Add
    ActionDef("add_camera", "Add &camera…", "Ctrl+Shift+C", "add_camera", icon="SP_MediaPlay",
              tip="Add a webcam, or a video / image file for testing"),
    ActionDef("add_projector", "Add &projector…", "Ctrl+Shift+P", "add_projector", icon="SP_ComputerIcon",
              tip="Add a projector: a display or a test window"),
    # View
    ActionDef("view_ornament", "&Ornament", "Ctrl+1", "view_mode", True, True, "view"),
    ActionDef("view_input", "Stitched &input", "Ctrl+2", "view_mode", True, False, "view"),
    ActionDef("view_labels", "&Labels", "Ctrl+3", "view_mode", True, False, "view"),
    ActionDef("view_cameras", "&Camera images", "Ctrl+4", "view_mode", True, False, "view",
              tip="Raw camera frames over the stage, for lining cameras up"),
    ActionDef("show_cameras", "Show cameras", "", "view_toggles", True, True),
    ActionDef("show_projectors", "Show projectors", "", "view_toggles", True, True),
    ActionDef("show_grid", "Show cell grid", "", "view_toggles", True, True),
    ActionDef("snap", "Snap to edges", "Ctrl+;", "view_toggles", True, True,
              tip="Hold Ctrl while dragging to place freely"),
    ActionDef("fit", "&Fit to window", "Ctrl+0", "view_fit"),
    ActionDef("zoom_in", "Zoom in", "Ctrl+=", "view_zoom_in"),
    ActionDef("zoom_out", "Zoom out", "Ctrl+-", "view_zoom_out"),
    # Output
    ActionDef("outputs", "Projector &outputs", "F5", "output_toggle", True, False, icon="SP_DesktopIcon",
              tip="Show every projector's image on its display / window"),
    ActionDef("test_pattern", "&Test pattern", "F6", "output_test_pattern", True, False,
              tip="Grid, names and overlap zones on the projectors, for physical alignment"),
    ActionDef("identify", "&Identify projectors", "F7", "output_identify",
              tip="Show each projector's number for a few seconds"),
    ActionDef("blackout", "&Blackout", "Ctrl+B", "output_blackout", True, False),
    # Run
    ActionDef("pause", "&Pause processing", "F8", "run_pause", True, False, icon="SP_MediaPause"),
    # Help
    ActionDef("shortcuts", "&Keyboard shortcuts", "F1", "help_shortcuts"),
    ActionDef("about", "&About", "", "help_about"),
)

MENUS = (
    ("&File", ("new", "open", "recent", "-", "save", "save_as", "-", "export", "-", "quit")),
    ("&Edit", ("undo", "redo", "-", "duplicate", "delete", "-", "select_stage")),
    ("&Add", ("add_camera", "add_projector")),
    ("&View", ("view_ornament", "view_input", "view_labels", "view_cameras", "-", "show_cameras",
               "show_projectors", "show_grid", "snap", "-", "fit", "zoom_in", "zoom_out", "-", "panels")),
    ("&Output", ("outputs", "test_pattern", "identify", "blackout")),
    ("&Run", ("pause",)),
    ("&Help", ("shortcuts", "about")),
)

TOOLBAR = ("new", "open", "save", "-", "undo", "redo", "-", "add_camera", "add_projector", "-",
           "view_mode_combo", "fit", "snap", "-", "pause", "outputs", "test_pattern")


def build_actions(window) -> dict[str, QAction]:
    actions: dict[str, QAction] = {}
    groups: dict[str, QActionGroup] = {}
    style = window.style()
    for d in ACTIONS:
        a = QAction(d.text, window)
        a.setObjectName(d.name)
        if d.shortcut:
            a.setShortcut(QKeySequence(d.shortcut))
        if d.tip:
            a.setToolTip(d.tip)
            a.setStatusTip(d.tip)
        if d.icon:
            a.setIcon(style.standardIcon(getattr(QStyle.StandardPixmap, d.icon)))
        if d.checkable:
            a.setCheckable(True)
            a.setChecked(d.checked)
        if d.group:
            groups.setdefault(d.group, QActionGroup(window)).addAction(a)
        handler = getattr(window, d.handler)
        if d.checkable:
            a.toggled.connect(lambda on, h=handler, n=d.name: h(n, on))
        else:
            a.triggered.connect(lambda checked=False, h=handler: h())
        actions[d.name] = a
    undo = window.doc.undo.createUndoAction(window, "&Undo")
    undo.setShortcut(QKeySequence.StandardKey.Undo)
    undo.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
    redo = window.doc.undo.createRedoAction(window, "&Redo")
    redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
    redo.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
    actions["undo"], actions["redo"] = undo, redo
    return actions
