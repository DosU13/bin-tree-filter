"""Per-frame processing for a whole scene: cameras -> stitched stage -> labels -> tree.

    Pipeline        does one frame per step(dt); owns sources, compositor, tracker, tree
    PipelineRunner  runs a Pipeline on a background thread; the UI / outputs read
                    latest() and send changes with set_scene() / set_params()

The result of a step (FrameResult) carries a TreeSnapshot for the renderers plus
small preview images and status for the UI.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from core import params as P
from core.compositor import Compositor
from core.engine import OrnamentTree, TreeSnapshot
from core.filters import apply_blur, apply_filters, apply_sharpen, hex_to_bgr
from core.posterize import PaletteTracker
from core.scene import Scene
from core.sources import open_scene_source

log = logging.getLogger(__name__)

# A camera whose newest frame is older than this no longer contributes
HOLD_SECONDS = 5.0


@dataclass
class FrameResult:
    snapshot: TreeSnapshot
    stage_input: np.ndarray          # stitched cameras at proc resolution (BGR)
    labels_image: np.ndarray         # posterized / band image at proc resolution (BGR)
    covered: np.ndarray              # bool, proc resolution
    timings: dict[str, float] = field(default_factory=dict)
    camera_status: dict[str, str] = field(default_factory=dict)
    fps: float = 0.0
    leaves: int = 0
    thumbnails: dict[str, np.ndarray] = field(default_factory=dict)   # small BGR frame per camera


def apply_params(values: dict, tree: OrnamentTree, tracker: PaletteTracker) -> None:
    """Push look parameters into the tree and the posterizer."""
    tree.max_depth = min(values["max_depth"], tree.max_level)
    tree.min_depth = min(values["min_depth"], tree.max_depth - 1)
    tree.threshold = values["threshold"]
    tree.hysteresis = values["hysteresis"]
    tree.color_hysteresis = values["color_hysteresis"]
    tree.conforming = values["conforming"]
    tree.animate = values["animate"]
    tree.split_time = values["split_time"]
    tree.tau_hist = values["tau_hist"]
    tree.tau_color = values["tau_color"]
    if tracker.k != values["colors"]:
        tracker.k = values["colors"]
        tracker.reset()
    tracker.style = values["palette_style"]
    tracker.tau = values["palette_tau"]
    tracker.label_margin = values["label_margin"]


def _source_key(cam) -> tuple:
    s = cam.source
    return (s.kind, s.index, s.device_name, s.backend, s.path, s.width, s.height, s.fps, s.fourcc, s.noise)


class Pipeline:
    """One scene's processing, one frame per step()."""

    def __init__(self, scene: Scene, realtime: bool = True) -> None:
        self.realtime = realtime
        self.scene = scene.copy()
        self.params = P.normalise(self.scene.params)
        self.sources: dict[str, object] = {}
        self._source_keys: dict[str, tuple] = {}
        self.tracker = PaletteTracker(k=self.params["colors"])
        self.tree: OrnamentTree | None = None
        self.compositor: Compositor | None = None
        self.version = 0
        self._last_time: dict[str, float] = {}
        self._last_id: dict[str, int] = {}
        self._rebuild_stage()
        self._sync_sources()

    # -- configuration -------------------------------------------------------

    def _rebuild_stage(self) -> None:
        st = self.scene.stage
        proc = self.params["proc_size"]
        self.tree = OrnamentTree(st.cols, st.rows, st.levels, proc)
        self.compositor = Compositor(st, proc)
        self.tracker.reset()
        apply_params(self.params, self.tree, self.tracker)

    def _sync_sources(self) -> None:
        wanted = {c.id: c for c in self.scene.cameras if c.enabled}
        for cam_id in list(self.sources):
            if cam_id not in wanted or self._source_keys[cam_id] != _source_key(wanted[cam_id]):
                self.sources.pop(cam_id).stop()
                self._source_keys.pop(cam_id)
                self._last_time.pop(cam_id, None)
                self._last_id.pop(cam_id, None)
        for cam_id, cam in wanted.items():
            if cam_id not in self.sources:
                self.sources[cam_id] = open_scene_source(cam.source, self.scene.resolve, name=cam.name or cam_id,
                                                         realtime=self.realtime)
                self._source_keys[cam_id] = _source_key(cam)

    def set_scene(self, scene: Scene) -> None:
        """Adopt an edited scene: only what changed is rebuilt / reopened."""
        old = self.scene
        self.scene = scene.copy()
        new_params = P.normalise(self.scene.params)
        stage_changed = old.stage != self.scene.stage or new_params["proc_size"] != self.params["proc_size"]
        self.params = new_params
        if stage_changed:
            self._rebuild_stage()
        else:
            apply_params(self.params, self.tree, self.tracker)
            old_cams = {c.id: c for c in old.cameras}
            for cam in self.scene.cameras:
                o = old_cams.get(cam.id)
                if o is None or o.transform != cam.transform or o.feather != cam.feather:
                    self.compositor.remove(cam.id)
            for cam_id in set(old_cams) - {c.id for c in self.scene.cameras}:
                self.compositor.remove(cam_id)
        self._sync_sources()

    def set_params(self, values: dict) -> None:
        values = P.normalise(values)
        proc_changed = values["proc_size"] != self.params["proc_size"]
        self.params = values
        self.scene.params = dict(values)
        if proc_changed:
            self._rebuild_stage()
        else:
            apply_params(values, self.tree, self.tracker)

    # -- per frame -----------------------------------------------------------

    def step(self, dt: float) -> FrameResult:
        t0 = time.perf_counter()
        timings = {}
        frames, status = {}, {}
        now = time.perf_counter()
        for cam in self.scene.cameras:
            if not cam.enabled:
                status[cam.id] = "disabled"
                continue
            src = self.sources.get(cam.id)
            frame, fid = src.latest()
            if fid != self._last_id.get(cam.id):
                self._last_id[cam.id] = fid
                self._last_time[cam.id] = now
            fresh = now - self._last_time.get(cam.id, -1e9) < HOLD_SECONDS
            status[cam.id] = src.status
            if frame is not None and fresh:
                frames[cam.id] = frame
        timings["capture"] = (time.perf_counter() - t0) * 1000

        p = self.params
        cams = self.scene.cameras
        t = time.perf_counter()
        if p["mode"] == "depth":
            band, covered = self.compositor.compose_band(cams, frames, p["depth_min"], p["depth_max"])
            timings["stitch"] = (time.perf_counter() - t) * 1000
            labels = (band > 0.5).astype(np.int64)
            palette = np.array([hex_to_bgr(p["depth_bg"]), hex_to_bgr(p["depth_fg"])], np.float32)
            stage_input = np.repeat((band * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        else:
            img, covered = self.compositor.compose(cams, frames)
            timings["stitch"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            img = apply_blur(img, p["blur"])
            img = apply_sharpen(img, p["sharpen"])
            stage_input = apply_filters(img, p["brightness"], p["contrast"], p["saturation"])
            timings["filter"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            labels, palette = self.tracker.update(stage_input, dt, mask=covered)
            # Uncovered pixels get an extra label drawn in the background colour (no ink)
            k = len(palette)
            labels = np.where(covered, labels, k)
            palette = np.vstack([palette, np.array(hex_to_bgr(p["background"]), np.float32)[None]])
            timings["posterize"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        self.tree.update(labels, palette, dt)
        timings["tree"] = (time.perf_counter() - t) * 1000
        self.version += 1
        snap = self.tree.snapshot()
        snap.version = self.version
        labels_image = palette[np.minimum(labels, len(palette) - 1)].astype(np.uint8)
        timings["total"] = (time.perf_counter() - t0) * 1000
        thumbs = {cid: img for cid, img in self.compositor.last_small.items() if cid in frames}
        return FrameResult(snap, stage_input, labels_image, covered, timings, status,
                           leaves=self.tree.leaf_count, thumbnails=thumbs)

    def close(self) -> None:
        for src in self.sources.values():
            src.stop()
        self.sources.clear()


class PipelineRunner:
    """Runs a Pipeline continuously on a background thread.

    latest() returns the newest FrameResult (or None); changes are queued and
    applied between frames, so the caller never blocks on processing.
    """

    def __init__(self, scene: Scene, max_fps: float = 60.0, realtime: bool = True) -> None:
        self.max_fps = max_fps
        self.realtime = realtime
        self._scene = scene
        self._latest: FrameResult | None = None
        self._lock = threading.Lock()
        self._commands: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.errors = 0
        self.last_error = ""
        self.fps = 0.0
        self.paused = False
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)

    def start(self) -> "PipelineRunner":
        self._thread.start()
        return self

    def latest(self) -> FrameResult | None:
        with self._lock:
            return self._latest

    def set_scene(self, scene: Scene) -> None:
        self._commands.put(("scene", scene.copy()))

    def set_params(self, values: dict) -> None:
        self._commands.put(("params", dict(values)))

    def _apply_commands(self, pipe: Pipeline) -> None:
        # Coalesce: only the newest scene / params matter
        scene = params = None
        while True:
            try:
                kind, value = self._commands.get_nowait()
            except queue.Empty:
                break
            if kind == "scene":
                scene, params = value, None
            else:
                params = value
        if scene is not None:
            pipe.set_scene(scene)
        if params is not None:
            pipe.set_params(params)

    def _run(self) -> None:
        try:
            pipe = Pipeline(self._scene, realtime=self.realtime)
        except Exception:
            log.exception("pipeline failed to start")
            self.last_error = "pipeline failed to start"
            return
        prev = time.perf_counter()
        frames, t_fps = 0, prev
        try:
            while not self._stop.is_set():
                start = time.perf_counter()
                try:
                    self._apply_commands(pipe)
                    if not self.paused:
                        dt = min(max(start - prev, 1 / 240), 0.25)
                        result = pipe.step(dt)
                        frames += 1
                        if start - t_fps >= 0.5:
                            self.fps = frames / (start - t_fps)
                            frames, t_fps = 0, start
                        result.fps = self.fps
                        with self._lock:
                            self._latest = result
                    prev = start
                except Exception as exc:          # keep running: one bad frame must not stop a show
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    log.exception("pipeline step failed")
                    self._stop.wait(0.1)
                budget = 1.0 / self.max_fps - (time.perf_counter() - start)
                if budget > 0:
                    self._stop.wait(budget)
        finally:
            pipe.close()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
