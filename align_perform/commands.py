"""Undoable edits.  Every change to the scene goes through one of these.

Commands call the document's underscore methods (which change the scene and
emit signals) and never touch the UI directly.
"""

from __future__ import annotations

import copy

from PySide6.QtGui import QUndoCommand

# Command ids for QUndoCommand.mergeWith (consecutive edits with the same merge
# key collapse into one undo step, e.g. a slider drag)
MERGE_ITEM = 1001
MERGE_PARAMS = 1002
MERGE_STAGE = 1003


class AddItemCommand(QUndoCommand):
    def __init__(self, doc, kind: str, spec, index: int | None = None, text: str = "") -> None:
        super().__init__(text or f"Add {kind} {spec.name or spec.id}")
        self.doc, self.kind, self.spec, self.index = doc, kind, copy.deepcopy(spec), index

    def redo(self) -> None:
        self.doc._insert_item(self.kind, copy.deepcopy(self.spec), self.index)
        self.doc.select(self.kind, self.spec.id)

    def undo(self) -> None:
        self.doc._remove_item(self.kind, self.spec.id)


class RemoveItemCommand(QUndoCommand):
    def __init__(self, doc, kind: str, item_id: str) -> None:
        spec = doc.scene.item(kind, item_id)
        super().__init__(f"Delete {kind} {spec.name or spec.id}")
        self.doc, self.kind, self.id = doc, kind, item_id
        self.spec = copy.deepcopy(spec)
        items = doc.scene.cameras if kind == "camera" else doc.scene.projectors
        self.index = [i.id for i in items].index(item_id)

    def redo(self) -> None:
        self.doc._remove_item(self.kind, self.id)

    def undo(self) -> None:
        self.doc._insert_item(self.kind, copy.deepcopy(self.spec), self.index)
        self.doc.select(self.kind, self.id)


class ReplaceItemCommand(QUndoCommand):
    """Replace a camera / projector spec (move, resize, property edit)."""

    def __init__(self, doc, kind: str, old, new, text: str, merge_key: str | None = None) -> None:
        super().__init__(text)
        self.doc, self.kind = doc, kind
        self.old, self.new = copy.deepcopy(old), copy.deepcopy(new)
        self.merge_key = merge_key

    def id(self) -> int:
        return MERGE_ITEM if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, ReplaceItemCommand) or other.merge_key != self.merge_key \
                or other.new.id != self.new.id or other.kind != self.kind:
            return False
        self.new = other.new
        return True

    def redo(self) -> None:
        self.doc._replace_item(self.kind, copy.deepcopy(self.new))

    def undo(self) -> None:
        self.doc._replace_item(self.kind, copy.deepcopy(self.old))


class StageCommand(QUndoCommand):
    def __init__(self, doc, old, new, text: str = "Edit stage", merge_key: str | None = None) -> None:
        super().__init__(text)
        self.doc, self.old, self.new, self.merge_key = doc, copy.deepcopy(old), copy.deepcopy(new), merge_key

    def id(self) -> int:
        return MERGE_STAGE if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, StageCommand) or other.merge_key != self.merge_key:
            return False
        self.new = other.new
        return True

    def redo(self) -> None:
        self.doc._set_stage(copy.deepcopy(self.new))

    def undo(self) -> None:
        self.doc._set_stage(copy.deepcopy(self.old))


class ParamsCommand(QUndoCommand):
    """Change look parameters; a run of changes to the same key merges."""

    def __init__(self, doc, old: dict, new: dict, text: str, merge_key: str | None = None) -> None:
        super().__init__(text)
        self.doc, self.old, self.new, self.merge_key = doc, dict(old), dict(new), merge_key

    def id(self) -> int:
        return MERGE_PARAMS if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, ParamsCommand) or other.merge_key != self.merge_key:
            return False
        self.new = other.new
        return True

    def redo(self) -> None:
        self.doc._set_params(dict(self.new))

    def undo(self) -> None:
        self.doc._set_params(dict(self.old))


class LooksCommand(QUndoCommand):
    def __init__(self, doc, old: dict, new: dict, text: str) -> None:
        super().__init__(text)
        self.doc, self.old, self.new = doc, copy.deepcopy(old), copy.deepcopy(new)

    def redo(self) -> None:
        self.doc._set_looks(copy.deepcopy(self.new))

    def undo(self) -> None:
        self.doc._set_looks(copy.deepcopy(self.old))
