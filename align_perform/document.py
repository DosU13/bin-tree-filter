"""SceneDocument: the scene being edited, its undo stack, selection and file state.

UI code reads `doc.scene` freely but changes it only through the public
methods here, which push undo commands.  The underscore methods are the
primitive edits the commands call; each emits the matching signal.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoStack

from align_perform.commands import (AddItemCommand, LooksCommand, ParamsCommand, RemoveItemCommand,
                                    ReplaceItemCommand, StageCommand)
from core import params as P
from core.scene import Scene, Stage

log = logging.getLogger(__name__)

KINDS = ("camera", "projector")
AUTOSAVE_SECONDS = 60


def new_scene() -> Scene:
    """Empty scene: a 2 x 1 stage of 1080 px cells, no cameras or projectors."""
    return Scene(stage=Stage(x=0, y=0, cols=2, rows=1, cell_size=1080, levels=12))


def autosave_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".autosave.json"


class SceneDocument(QObject):
    sceneChanged = Signal()            # anything changed (pipeline sync, title)
    itemsChanged = Signal()            # cameras / projectors added or removed
    itemChanged = Signal(str, str)     # kind, id: properties or placement
    stageChanged = Signal()
    paramsChanged = Signal()
    looksChanged = Signal()
    selectionChanged = Signal()
    fileChanged = Signal()             # path or dirty state

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.scene = new_scene()
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(lambda _: self.fileChanged.emit())
        self.selection: tuple[str, str] | None = None
        self._autosave = QTimer(self)
        self._autosave.setInterval(AUTOSAVE_SECONDS * 1000)
        self._autosave.timeout.connect(self.autosave)
        self._autosave.start()

    # -- state ---------------------------------------------------------------

    @property
    def path(self) -> str | None:
        return self.scene.path

    @property
    def dirty(self) -> bool:
        return not self.undo.isClean()

    def title(self) -> str:
        name = os.path.basename(self.path) if self.path else "Untitled"
        return name + (" *" if self.dirty else "")

    # -- selection -----------------------------------------------------------

    def select(self, kind: str | None, item_id: str = "") -> None:
        sel = None if kind is None else (kind, item_id)
        if sel != self.selection:
            self.selection = sel
            self.selectionChanged.emit()

    def selected_spec(self):
        if not self.selection or self.selection[0] not in KINDS:
            return None
        try:
            return self.scene.item(*self.selection)
        except KeyError:
            return None

    # -- primitive edits (commands call these) --------------------------------

    def _items(self, kind: str) -> list:
        return self.scene.cameras if kind == "camera" else self.scene.projectors

    def _insert_item(self, kind: str, spec, index: int | None) -> None:
        items = self._items(kind)
        items.insert(len(items) if index is None else index, spec)
        self.itemsChanged.emit()
        self.sceneChanged.emit()

    def _remove_item(self, kind: str, item_id: str) -> None:
        items = self._items(kind)
        items[:] = [i for i in items if i.id != item_id]
        if self.selection == (kind, item_id):
            self.select(None)
        self.itemsChanged.emit()
        self.sceneChanged.emit()

    def _replace_item(self, kind: str, spec) -> None:
        items = self._items(kind)
        for i, item in enumerate(items):
            if item.id == spec.id:
                items[i] = spec
                break
        self.itemChanged.emit(kind, spec.id)
        self.sceneChanged.emit()

    def _set_stage(self, stage: Stage) -> None:
        self.scene.stage = stage
        self.stageChanged.emit()
        self.sceneChanged.emit()

    def _set_params(self, values: dict) -> None:
        self.scene.params = P.normalise(values)
        self.paramsChanged.emit()
        self.sceneChanged.emit()

    def _set_looks(self, looks: dict) -> None:
        self.scene.looks = looks
        self.looksChanged.emit()
        self.sceneChanged.emit()

    # -- editing API ---------------------------------------------------------

    def add_item(self, kind: str, spec) -> None:
        if not spec.id:
            spec.id = self.scene.new_id("cam" if kind == "camera" else "proj")
        self.undo.push(AddItemCommand(self, kind, spec))

    def remove_item(self, kind: str, item_id: str) -> None:
        self.undo.push(RemoveItemCommand(self, kind, item_id))

    def duplicate_item(self, kind: str, item_id: str, offset: float = 40.0) -> None:
        spec = copy.deepcopy(self.scene.item(kind, item_id))
        spec.id = self.scene.new_id("cam" if kind == "camera" else "proj")
        spec.name = (spec.name or item_id) + " copy"
        t = spec.transform
        t.x += offset
        t.y += offset
        if t.corners is not None:
            t.corners = [[x + offset, y + offset] for x, y in t.corners]
        self.add_item(kind, spec)

    def update_item(self, kind: str, item_id: str, mutate: Callable, text: str,
                    merge_key: str | None = None) -> None:
        """Apply *mutate(spec_copy)* as one undoable edit (merged with the previous one
        when both use the same *merge_key*, e.g. a drag or a slider)."""
        old = self.scene.item(kind, item_id)
        new = copy.deepcopy(old)
        mutate(new)
        if new == old:
            return
        self.undo.push(ReplaceItemCommand(self, kind, old, new, text, merge_key))

    def set_item(self, kind: str, spec, text: str, merge_key: str | None = None) -> None:
        old = self.scene.item(kind, spec.id)
        if spec != old:
            self.undo.push(ReplaceItemCommand(self, kind, old, spec, text, merge_key))

    def update_derived(self, kind: str, item_id: str, mutate: Callable) -> None:
        """Change information derived from the hardware (e.g. a detected frame size)
        without an undo step: undoing it would only make it come straight back."""
        spec = copy.deepcopy(self.scene.item(kind, item_id))
        mutate(spec)
        if spec != self.scene.item(kind, item_id):
            self._replace_item(kind, spec)

    def set_stage(self, stage: Stage, text: str = "Edit stage", merge_key: str | None = None) -> None:
        if stage != self.scene.stage:
            self.undo.push(StageCommand(self, self.scene.stage, stage, text, merge_key))

    def set_params(self, values: dict, text: str = "Change look", merge_key: str | None = None) -> None:
        values = P.normalise(values)
        if values != self.scene.params:
            self.undo.push(ParamsCommand(self, self.scene.params, values, text, merge_key))

    def set_param(self, key: str, value) -> None:
        values = dict(self.scene.params)
        values[key] = P.coerce(key, value)
        self.set_params(values, f"{P.BY_KEY[key].label}", merge_key=key)

    def nudge_param(self, key: str, direction: int) -> None:
        self.set_param(key, P.nudge(self.scene.params, key, direction))

    # -- looks ---------------------------------------------------------------

    def save_look(self, name: str) -> None:
        looks = dict(self.scene.looks)
        looks[name] = P.look_values(self.scene.params)        # show settings stay out of looks
        self.undo.push(LooksCommand(self, self.scene.looks, looks, f"Save look {name}"))

    def delete_look(self, name: str) -> None:
        looks = {k: v for k, v in self.scene.looks.items() if k != name}
        self.undo.push(LooksCommand(self, self.scene.looks, looks, f"Delete look {name}"))

    def rename_look(self, old: str, new: str) -> None:
        if not new or new == old or new in self.scene.looks:
            return
        looks = {(new if k == old else k): v for k, v in self.scene.looks.items()}
        self.undo.push(LooksCommand(self, self.scene.looks, looks, f"Rename look {old}"))

    def apply_look(self, name: str) -> None:
        if name in self.scene.looks:
            self.set_params(P.look_target(self.scene.params, self.scene.looks[name]), f"Apply look {name}")

    # -- files ---------------------------------------------------------------

    def _reset(self, scene: Scene) -> None:
        self.scene = scene
        self.undo.clear()
        self.selection = None
        self.itemsChanged.emit()
        self.stageChanged.emit()
        self.paramsChanged.emit()
        self.looksChanged.emit()
        self.selectionChanged.emit()
        self.sceneChanged.emit()
        self.fileChanged.emit()

    def new(self) -> None:
        self._reset(new_scene())

    def load(self, path: str, from_autosave: bool = False) -> None:
        """Load a scene file (raises core.scene.SceneError on bad files)."""
        src = autosave_path(path) if from_autosave else path
        scene = Scene.load(src)
        scene.path = os.path.abspath(path)
        self._reset(scene)
        if from_autosave:
            # The restored content differs from the file on disk: mark dirty
            self.undo.resetClean()
            self.fileChanged.emit()

    def save(self, path: str | None = None) -> None:
        self.scene.save(path)
        self.undo.setClean()
        try:
            os.remove(autosave_path(self.scene.path))
        except OSError:
            pass
        self.fileChanged.emit()

    def autosave(self) -> None:
        if self.dirty and self.path:
            try:
                copy.deepcopy(self.scene).save(autosave_path(self.path))
            except Exception as exc:
                log.warning("autosave failed: %s", exc)

    def has_newer_autosave(self, path: str) -> bool:
        a = autosave_path(path)
        return os.path.exists(a) and os.path.getmtime(a) > os.path.getmtime(path) + 1
