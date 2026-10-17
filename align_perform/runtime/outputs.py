"""ProjectorOutputs: one ProjectorWindow per enabled projector, kept in step with the scene.

Used by Align Perform (Output menu) and perform.py.  Places each window full
screen on its display (matched by serial, name, index) or as a test window,
repaints them at the display rate, and handles the output-level controls:
test pattern, identify, blackout fade, HUD, and arrow-key nudging of the
projector (or one 4-corner warp corner) while looking at the projection.
"""

from __future__ import annotations

import copy
import logging
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication

from align_perform.runtime.projector_window import ProjectorWindow
from align_perform.runtime.screens import find_screen

log = logging.getLogger(__name__)


class ProjectorOutputs(QObject):
    statesChanged = Signal(dict)                    # projector id -> shown | missing | hidden | off
    closedByUser = Signal()
    adjustRequested = Signal(str, object, float, float)   # id, corner index or None, dx, dy (world px)

    def __init__(self, scene, session, key_handler=None, windowed: bool = False,
                 missing: str = "window", fade_seconds: float = 0.6, parent: QObject | None = None) -> None:
        """
        Args:
            session:     object with .tile (ornament tile array) and .last (FrameResult)
            key_handler: callable(QKeyEvent) for keys the outputs don't use themselves
            windowed:    show every projector as a window (testing)
            missing:     what to do when a projector's display isn't connected:
                         "window" (show a window instead) or "wait" (stay hidden, retry)
        """
        super().__init__(parent)
        self.session = session
        self.key_handler = key_handler
        self.windowed = windowed
        self.missing_policy = missing
        self.fade_seconds = fade_seconds
        self.scene = scene.copy()
        self.params = dict(self.scene.params)
        self.windows: dict[str, ProjectorWindow] = {}
        self.visible = False
        self.test_pattern = False
        self.blackout = False
        self.level = 1.0
        self.identify_until = 0.0
        self.hud_lines: list[str] = []
        self.corner_sel: dict[str, int | None] = {}
        self.geometry_version = 0
        self.snapshot = None
        self.closing = False
        self.allow_escape = True
        self._states: dict[str, str] = {}
        self._last_tick = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._retry = QTimer(self)
        self._retry.setInterval(2000)
        self._retry.timeout.connect(self._retry_missing)
        self._retry.start()
        self._sync()

    # -- data for windows ----------------------------------------------------

    @property
    def tile(self):
        return getattr(self.session, "tile", None)

    def projector(self, proj_id: str):
        try:
            return self.scene.projector(proj_id)
        except KeyError:
            return None

    def number(self, proj_id: str) -> int:
        ids = [p.id for p in self.scene.projectors]
        return ids.index(proj_id) + 1 if proj_id in ids else 0

    # -- scene ---------------------------------------------------------------

    def set_scene(self, scene) -> None:
        self.scene = scene.copy()
        self.params = dict(self.scene.params)
        self.geometry_version += 1
        self._sync()

    def preview_params(self, values: dict) -> None:
        self.params = dict(values)

    def push(self, result) -> None:
        self.snapshot = result.snapshot

    def _sync(self) -> None:
        wanted = {p.id: p for p in self.scene.projectors if p.enabled}
        for pid in list(self.windows):
            if pid not in wanted:
                self._destroy(pid)
        first = True
        for pid, proj in wanted.items():
            win = self.windows.get(pid)
            if win is None:
                win = ProjectorWindow(self, pid, vsync=first)
                self.windows[pid] = win
            first = False
            if self.visible and win.placed_output != proj.output:
                self._place(win, proj)
            self._retitle(win, proj)
        self._emit_states()

    def _retitle(self, win: ProjectorWindow, proj) -> None:
        missing = "  (display not connected)" if win.state == "missing" else ""
        win.setTitle(f"{self.number(proj.id)}: {proj.name or proj.id} — "
                     f"{proj.output.width}×{proj.output.height}{missing}")

    def _destroy(self, pid: str) -> None:
        win = self.windows.pop(pid)
        self.closing = True
        try:
            win.release_gl()
            win.close()
            win.deleteLater()
        finally:
            self.closing = False
        self._states.pop(pid, None)

    def _place(self, win: ProjectorWindow, proj) -> None:
        out = proj.output
        win.placed_output = copy.deepcopy(out)
        if out.kind == "none":
            win.hide()
            win.state = "off"
            return
        screen = None if (self.windowed or out.kind != "screen") else find_screen(out)
        if out.kind == "screen" and not self.windowed and screen is None:
            log.warning("projector %s: display %s not connected", proj.id, out.screen_name or out.screen_index)
            if self.missing_policy == "wait":
                win.hide()
                win.state = "missing"
                win.placed_output = None          # retry later
                return
            win.state = "missing"
            self._show_window(win, proj)
            self._retitle(win, proj)
            return
        if screen is not None:
            win.setFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
            win.setScreen(screen)
            win.setGeometry(screen.geometry())
            win.showFullScreen()
            win.hide_cursor(True)
            win.state = "shown"
        else:
            self._show_window(win, proj)
            win.state = "shown"

    def _show_window(self, win: ProjectorWindow, proj) -> None:
        out = proj.output
        win.setFlags(Qt.WindowType.Window)
        win.hide_cursor(False)
        if not win.isVisible():
            avail = QGuiApplication.primaryScreen().availableGeometry()
            w = min(out.width, int(avail.width() * 0.42))
            h = max(64, int(w * out.height / out.width))
            if h > avail.height() * 0.6:
                h = int(avail.height() * 0.6)
                w = int(h * out.width / out.height)
            n = self.number(proj.id) - 1
            win.resize(w, h)
            win.setPosition(avail.x() + 40 + 50 * n, avail.y() + 60 + 50 * n)
        win.showNormal()

    def _retry_missing(self) -> None:
        if not self.visible:
            return
        for pid, win in self.windows.items():
            if win.state == "missing" and self.missing_policy == "wait":
                proj = self.projector(pid)
                if proj is not None and find_screen(proj.output) is not None:
                    log.info("projector %s: display connected", pid)
                    self._place(win, proj)
        self._emit_states()

    def _emit_states(self) -> None:
        states = {}
        for p in self.scene.projectors:
            win = self.windows.get(p.id)
            if not p.enabled:
                states[p.id] = "disabled"
            elif win is None or not self.visible:
                states[p.id] = "hidden"
            else:
                states[p.id] = win.state
        if states != self._states:
            self._states = states
            self.statesChanged.emit(states)

    # -- controls ------------------------------------------------------------

    def set_visible(self, on: bool) -> None:
        self.visible = on
        for pid, win in self.windows.items():
            if on:
                self._place(win, self.scene.projector(pid))
            else:
                win.hide()
                win.placed_output = None
                win.state = "hidden"
        self._emit_states()

    def set_test_pattern(self, on: bool) -> None:
        self.test_pattern = on

    def identify(self, seconds: float = 4.0) -> None:
        self.identify_until = time.perf_counter() + seconds

    def set_blackout(self, on: bool) -> None:
        self.blackout = on

    def set_hud(self, lines: list[str]) -> None:
        self.hud_lines = list(lines)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        target = 0.0 if self.blackout else 1.0
        if self.level != target:
            step = dt / max(self.fade_seconds, 1e-3)
            self.level = min(target, self.level + step) if target > self.level else max(target, self.level - step)
        if not self.visible:
            return
        for win in self.windows.values():
            if win.isVisible():
                win.update()

    # -- input ---------------------------------------------------------------

    def handle_key(self, proj_id: str, event) -> None:
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Escape and self.allow_escape:
            self.closedByUser.emit()
            return
        if self.test_pattern:
            proj = self.projector(proj_id)
            if key == Qt.Key.Key_Tab and proj is not None and proj.transform.corners is not None:
                cur = self.corner_sel.get(proj_id)
                self.corner_sel[proj_id] = 0 if cur is None else (None if cur == 3 else cur + 1)
                return
            step = 10.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
            arrows = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                      Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
            if key in arrows:
                corner = self.corner_sel.get(proj_id) if proj is not None and proj.transform.corners is not None else None
                self.adjustRequested.emit(proj_id, corner, *arrows[key])
                return
        if self.key_handler is not None:
            self.key_handler(event)

    def window_closed(self, proj_id: str) -> None:
        self.closedByUser.emit()

    def states(self) -> dict[str, str]:
        return dict(self._states)

    def close(self) -> None:
        self._timer.stop()
        self._retry.stop()
        for pid in list(self.windows):
            self._destroy(pid)
