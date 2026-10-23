"""Play a scene on stage / at an installation: every projector full screen, no editor.

    python perform.py scenes/show.json
    python perform.py scenes/show.json --look calm --status
    python perform.py scenes/example.json --windowed          # try it on one monitor

Set the scene up in Align Perform (python -m align_perform), save it, play it here.

Keys (on any projector window, or the --status window):
    live.py keys        change the look live (d/D, h/H, c/C, g, x, ... ; not saved)
    1 - 9               crossfade to the scene's looks, in order (--fade seconds)
    0                   crossfade back to the scene's own settings
    B                   blackout (fades out / in)
    L                   learn the empty stage (for presence / waves / idle look; step out of view first)
    F                   freeze the picture
    R                   reload the scene file
    H                   status overlay on the projectors
    T                   test pattern        I   identify projectors
    S                   save each projector's image to the log folder
    Ctrl+Shift+Q        quit (Esc does nothing during a show)

Runs unattended: cameras that fail are reopened, a stalled or crashed pipeline
is restarted, missing displays are waited for (--wait-for-screens), errors are
logged to logs/perform-YYYYMMDD.log and the display is kept awake.  If the scene
has an idle look, it takes over while nobody is on stage.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import core  # noqa: F401  (OpenCV env before cv2 loads)
import cv2
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from align_perform.app import dark_palette
from align_perform.runtime.outputs import ProjectorOutputs
from align_perform.runtime.qt_gl import configure_opengl
from align_perform.runtime.session import LiveSession
from core import params as P
from core.looks import LookFader
from core.scene import Scene, SceneError

log = logging.getLogger("perform")

STALL_SECONDS = 10.0          # no new frame for this long (not frozen) -> restart the pipeline
HEALTH_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Process setup
# ---------------------------------------------------------------------------

class DailyFileHandler(logging.FileHandler):
    """Writes to <folder>/perform-YYYYMMDD.log, switching files at midnight."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.day = time.strftime("%Y%m%d")
        super().__init__(self._path(), encoding="utf-8")

    def _path(self) -> str:
        return os.path.join(self.folder, f"perform-{self.day}.log")

    def emit(self, record) -> None:
        day = time.strftime("%Y%m%d")
        if day != self.day:
            self.day = day
            self.close()
            self.baseFilename = os.path.abspath(self._path())
            self.stream = self._open()
        super().emit(record)


def setup_logging(log_dir: str, verbose: bool) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = DailyFileHandler(log_dir)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    def excepthook(kind, value, tb):
        log.critical("unhandled exception", exc_info=(kind, value, tb))

    def thread_excepthook(args):
        log.critical("unhandled exception in thread %s", args.thread.name if args.thread else "?",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook


def rss_mb() -> float:
    """This process's resident memory in MB (Windows; 0 elsewhere)."""
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE      # 64-bit handle, not a C int
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return c.WorkingSetSize / 1e6
    return 0.0


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping or blanking the displays while the show runs."""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------

class StatusWindow(QWidget):
    """Small operator window: health at a glance; also takes the show keys."""

    def __init__(self, show: "Show") -> None:
        super().__init__()
        self.show_ = show
        self.setWindowTitle("perform — status")
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        self.resize(460, 260)

    def keyPressEvent(self, event) -> None:
        self.show_.on_key(event)

    def closeEvent(self, event) -> None:
        event.ignore()            # quitting is Ctrl+Shift+Q only


class Show(QObject):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.path = os.path.abspath(args.scene)
        self.scene = Scene.load(self.path)
        self.base_params = dict(self.scene.params)
        self.params = dict(self.scene.params)
        self.look_name = "scene"
        if args.look:
            if args.look not in self.scene.looks:
                raise SceneError(f"look {args.look!r} not in scene (has: {', '.join(self.scene.looks) or 'none'})")
            self.params = P.look_target(self.params, self.scene.looks[args.look])
            self.look_name = args.look
        self.frozen = False
        self.fader: LookFader | None = None
        self.started = time.time()
        self.frames = 0
        self.restarts = 0
        self._last_version = -1
        self._last_new_frame = time.perf_counter()
        self._last_health = time.perf_counter()
        self.hud = False
        self._idle_state = "off"           # off | fading | on (for the log)

        self.session = self._new_session()
        self.outputs = ProjectorOutputs(self.scene, self.session, key_handler=self.on_key, windowed=args.windowed,
                                        missing="wait" if args.wait_for_screens else "window",
                                        fade_seconds=args.blackout_fade)
        self.outputs.allow_escape = False
        self.outputs.statesChanged.connect(lambda s: log.info("projectors: %s", s))
        self.outputs.preview_params(self.params)
        self.outputs.set_visible(True)

        self.status_window = StatusWindow(self) if args.status else None
        if self.status_window is not None:
            self.status_window.show()

        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(33)
        self._fade_timer.timeout.connect(self._fade_step)
        self._watch = QTimer(self)
        self._watch.setInterval(1000)
        self._watch.timeout.connect(self._watchdog)
        self._watch.start()
        if args.simulate_faults:
            self._faults = QTimer(self)
            self._faults.setInterval(int(args.simulate_faults * 1000))
            self._faults.timeout.connect(self._inject_fault)
            self._faults.start()
        if args.duration:
            QTimer.singleShot(int(args.duration * 1000), self.quit)
        log.info("show started: %s (%d cameras, %d projectors, look %s)", self.path,
                 len(self.scene.enabled_cameras()), len(self.scene.enabled_projectors()), self.look_name)

    # -- runtime -------------------------------------------------------------

    def _new_session(self) -> LiveSession:
        scene = self.scene.copy()
        scene.params = dict(self.params)
        session = LiveSession(scene, self, max_fps=self.args.max_fps)
        session.frameReady.connect(self._on_frame)
        session.message.connect(lambda m: log.warning("%s", m))
        return session

    def _on_frame(self, r) -> None:
        self.outputs.push(r)
        self.frames += 1
        if r.snapshot.version != self._last_version:
            self._last_version = r.snapshot.version
            self._last_new_frame = time.perf_counter()
        if self.hud:
            self.outputs.set_hud(self.status_lines(r))
        state = "off" if r.idle <= 0 else ("on" if r.idle >= 1 else "fading")
        if state != self._idle_state:
            if state == "on" or (state == "fading" and self._idle_state == "off"):
                log.info("idle look %s (nobody on stage)", "on" if state == "on" else "fading in")
            elif state == "off":
                log.info("idle look off (someone came)")
            self._idle_state = state

    def apply_params(self, values: dict) -> None:
        self.params = P.normalise(values)
        self.session.preview_params(self.params)
        self.outputs.preview_params(self.params)

    def status_lines(self, r=None) -> list[str]:
        r = r or self.session.last
        up = int(time.time() - self.started)
        idle = f"  idle {r.idle:.0%}" if r is not None and r.idle > 0 else ""
        lines = [f"{time.strftime('%H:%M:%S')}  up {up // 3600}:{up // 60 % 60:02d}:{up % 60:02d}  "
                 f"look {self.look_name}{idle}{'  FROZEN' if self.frozen else ''}"
                 f"{'  BLACKOUT' if self.outputs.blackout else ''}"]
        if r is not None:
            people = ""
            if r.activity is not None:
                people = "  learning stage" if r.learning else f"  people {r.people_count}"
            lines.append(f"{r.fps:.1f} fps  {r.timings.get('total', 0):.1f} ms  leaves {r.leaves}{people}  "
                         f"errors {self.session.runner.errors}  restarts {self.restarts}  mem {rss_mb():.0f} MB")
            cams = "  ".join(f"{(c.name or c.id)}:{r.camera_status.get(c.id, '-')}" for c in self.scene.cameras)
            lines.append(f"cameras  {cams or 'none'}")
        proj = "  ".join(f"{pid}:{st}" for pid, st in self.outputs.states().items())
        lines.append(f"projectors  {proj}")
        return lines

    def _watchdog(self) -> None:
        now = time.perf_counter()
        runner = self.session.runner
        dead = not runner._thread.is_alive()
        stalled = not self.frozen and now - self._last_new_frame > STALL_SECONDS
        if dead or stalled:
            self.restarts += 1
            log.error("pipeline %s; restarting (restart #%d)", "died" if dead else "stalled", self.restarts)
            old = self.session
            self.session = self._new_session()
            self.outputs.session = self.session
            self._last_new_frame = now
            try:
                old.close()
            except Exception:
                log.exception("closing old pipeline")
        if self.status_window is not None:
            self.status_window.label.setText("<br>".join(self.status_lines()))
        if now - self._last_health >= HEALTH_SECONDS:
            self._last_health = now
            log.info("health | %s", " | ".join(self.status_lines()))

    # -- controls ------------------------------------------------------------

    def crossfade(self, name: str | None) -> None:
        target = self.base_params if name is None else self.scene.looks.get(name)
        if target is None:
            return
        self.look_name = name or "scene"
        log.info("look -> %s (%.1f s)", self.look_name, self.args.fade)
        self.fader = LookFader(self.params, target, self.args.fade, time.perf_counter())
        self._fade_timer.start()

    def _fade_step(self) -> None:
        if self.fader is None:
            self._fade_timer.stop()
            return
        now = time.perf_counter()
        self.apply_params(self.fader.values(now))
        if self.fader.done(now):
            self.fader = None
            self._fade_timer.stop()

    def reload(self) -> None:
        try:
            scene = Scene.load(self.path)
        except (OSError, SceneError) as exc:
            log.error("reload failed: %s", exc)
            return
        self.scene = scene
        self.base_params = dict(scene.params)
        self.params = dict(scene.params)
        self.look_name = "scene"
        running = scene.copy()
        running.params = dict(self.params)
        self.session.set_scene(running)
        self.outputs.set_scene(running)
        log.info("scene reloaded")

    def snapshot(self) -> None:
        folder = os.path.join(self.args.log_dir, "snapshots", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        for p in self.scene.enabled_projectors():
            img = self.session.render_projector(p)
            if img is not None:
                cv2.imwrite(os.path.join(folder, f"{p.id}.png"), img)
        log.info("snapshot saved to %s", folder)

    def on_key(self, event) -> None:
        key, mods, text = event.key(), event.modifiers(), event.text()
        ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        if key == Qt.Key.Key_Q and (mods & ctrl_shift) == ctrl_shift:
            self.quit()
            return
        if key == Qt.Key.Key_Escape:
            log.info("Esc ignored during a show; Ctrl+Shift+Q quits")
            return
        if text and text in "123456789":
            names = list(self.scene.looks)
            i = int(text) - 1
            if i < len(names):
                self.crossfade(names[i])
            return
        if text == "0":
            self.crossfade(None)
            return
        upper = {"B": self._toggle_blackout, "F": self._toggle_freeze, "R": self.reload,
                 "H": self._toggle_hud, "T": self._toggle_pattern, "I": self.outputs.identify,
                 "S": self.snapshot, "L": self._relearn}
        if text in upper:
            upper[text]()
            return
        hot = P.hotkeys(self.params["mode"])
        if text in hot:
            param, direction = hot[text]
            values = dict(self.params)
            values[param] = P.nudge(values, param, direction)
            self.apply_params(values)
            log.info("%s -> %s", param, values[param])

    def _toggle_blackout(self) -> None:
        self.outputs.set_blackout(not self.outputs.blackout)
        log.info("blackout %s", "on" if self.outputs.blackout else "off")

    def _relearn(self) -> None:
        self.session.relearn_background()
        log.info("learning the empty stage")

    def _toggle_freeze(self) -> None:
        self.frozen = not self.frozen
        self.session.set_paused(self.frozen)
        self._last_new_frame = time.perf_counter()
        log.info("freeze %s", "on" if self.frozen else "off")

    def _toggle_hud(self) -> None:
        self.hud = not self.hud
        self.outputs.set_hud(self.status_lines() if self.hud else [])

    def _toggle_pattern(self) -> None:
        self.outputs.set_test_pattern(not self.outputs.test_pattern)

    def _inject_fault(self) -> None:
        """--simulate-faults: break things on purpose to prove the show survives."""
        sources = list(self.session.runner_sources())
        self._fault_count = getattr(self, "_fault_count", 0) + 1
        choice = ("camera", "camera", "pipeline")[self._fault_count % 3]
        if choice == "camera" and sources:
            name, src = sources[self._fault_count % len(sources)]
            log.warning("FAULT: closing camera %s under its reader", name)
            try:
                src.reader.close()
            except Exception:
                pass
        else:
            log.warning("FAULT: next pipeline step raises")
            self.session.runner.inject_fault = True

    def quit(self) -> None:
        log.info("quitting: %d frames shown, %d pipeline restarts, %d step errors", self.frames,
                 self.restarts, self.session.runner.errors)
        self._watch.stop()
        self.outputs.close()
        self.session.close()
        if self.status_window is not None:
            self.status_window.hide()
        QApplication.instance().quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene file made with Align Perform")
    ap.add_argument("--look", help="start with this look from the scene")
    ap.add_argument("--fade", type=float, default=3.0, help="look crossfade seconds (default 3)")
    ap.add_argument("--blackout-fade", type=float, default=1.0, help="blackout fade seconds (default 1)")
    ap.add_argument("--windowed", action="store_true", help="every projector in a window (testing)")
    ap.add_argument("--wait-for-screens", action="store_true",
                    help="keep a projector dark until its display is connected (default: show it in a window)")
    ap.add_argument("--status", action="store_true", help="show a small status window")
    ap.add_argument("--max-fps", type=float, default=60.0, help="processing rate cap (default 60)")
    ap.add_argument("--log-dir", default="logs", help="log folder (default logs)")
    ap.add_argument("--duration", type=float, default=0, help="quit after this many seconds (testing)")
    ap.add_argument("--simulate-faults", type=float, default=0, metavar="SECONDS",
                    help="testing: break a camera or a pipeline step every SECONDS")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_dir, args.verbose)
    configure_opengl(vsync=True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("perform")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setQuitOnLastWindowClosed(False)
    try:
        show = Show(args)
    except (OSError, SceneError) as exc:
        log.error("cannot start: %s", exc)
        return 2
    keep_awake(True)
    try:
        code = app.exec()
    finally:
        keep_awake(False)
    log.info("exit %d", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
