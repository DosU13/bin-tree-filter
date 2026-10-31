"""Per-frame processing for a whole scene: cameras -> stitched stage -> labels -> tree.

    Pipeline        does one frame per step(dt); owns sources, compositor, tracker, tree,
                    presence (people, waves) and the idle look
    PipelineRunner  runs a Pipeline on a background thread; the UI / outputs read
                    latest() and send changes with set_scene() / set_params()

The result of a step (FrameResult) carries a TreeSnapshot for the renderers plus
small preview images and status for the UI.

The settings in force are `Pipeline.effective`: the params sent in, faded towards
the idle look while nobody is on stage (FrameResult.idle says how far; renderers
mix their own style the same way with core.looks.idle_mix).
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
from core.engine import OrnamentTree, TreeSnapshot, load_tile, tile_edge_ink
from core.filters import apply_blur, apply_filters, apply_sharpen, hex_to_bgr
from core.glow import MotionGlow, glow_labels
from core.looks import IdleFader, idle_mix
from core.posterize import DepthLayers, PaletteTracker, depth_palette, split_main
from core.presence import Person, PresenceTracker, WaveEmitter
from core.scene import Scene
from core.sources import open_scene_source
from core.view import render_background

log = logging.getLogger(__name__)

# A camera whose newest frame is older than this no longer contributes
HOLD_SECONDS = 5.0
# Presence detection that was off longer than this relearns the empty stage first
PRESENCE_STALE_SECONDS = 1.0
# "Colours from people" needs at least this share of the stage moving
PALETTE_MIN_PEOPLE = 0.005


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
    frame_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)   # real (w, h) per camera
    activity: np.ndarray | None = None       # presence 0..1 at proc resolution (None: not running)
    people: list[Person] = field(default_factory=list)
    learning: bool = False                   # presence is (re)learning the empty stage
    idle: float = 0.0                        # how far the idle look is faded in (0..1)

    @property
    def people_count(self) -> int:
        return sum(1 for p in self.people if p.missing == 0)


def needs_presence(values: dict, idle: bool = True) -> bool:
    """Whether these settings use people detection at all (about 0.5 ms per cell a frame)."""
    return bool(values["presence"] or values["waves"] or (idle and values["idle_look"]))


def apply_params(values: dict, tree: OrnamentTree, tracker: PaletteTracker) -> None:
    """Push look parameters into the tree and the posterizer."""
    tree.max_depth = min(values["max_depth"], tree.max_level)
    tree.min_depth = min(values["min_depth"], tree.max_depth - 1)
    tree.threshold = values["threshold"]
    tree.hysteresis = values["hysteresis"]
    tree.color_hysteresis = values["color_hysteresis"]
    tree.conforming = values["conforming"]
    tree.polarity = values["polarity"]
    tree.animate = values["animate"]
    tree.split_time = values["split_time"]
    tree.tau_hist = values["tau_hist"]
    tree.tau_color = values["tau_color"]
    if tracker.k != values["colors"]:
        tracker.k = values["colors"]
        tracker.reset()
    tracker.style = values["palette_style"]
    tracker.glow = values["glow"]
    tracker.tau = values["palette_tau"]
    tracker.label_margin = values["label_margin"]
    tree.still_depth = values["still_depth"]
    tree.still_dim = values["still_dim"]
    tree.dim_color = render_background(values)


def glow_colour(values: dict) -> tuple[int, int, int]:
    """The motion glow's colour (BGR): the main colour of the palette in use, else Light colour."""
    split = split_main(values["depth_palette"] if values["mode"] == "depth" else values["palette_style"])
    return hex_to_bgr(split[1] if split else values["glow_color"])


def _source_key(cam) -> tuple:
    """Source settings that need the source reopened (an Astra's depth range and clean-up don't)."""
    s = cam.source
    return (s.kind, s.index, s.device_name, s.backend, s.path, s.width, s.height, s.fps, s.fourcc, s.noise,
            s.stream)


class Pipeline:
    """One scene's processing, one frame per step()."""

    def __init__(self, scene: Scene, realtime: bool = True) -> None:
        self.realtime = realtime
        self.scene = scene.copy()
        self.params = P.normalise(self.scene.params)
        self.sources: dict[str, object] = {}
        self._source_keys: dict[str, tuple] = {}
        self.tracker = PaletteTracker(k=self.params["colors"])
        self.layers = DepthLayers()
        self.tree: OrnamentTree | None = None
        self.compositor: Compositor | None = None
        self.presence = PresenceTracker()
        self.waves = WaveEmitter()
        self.glow = MotionGlow()
        self.idle = IdleFader()
        self.idle_enabled = True
        self.effective = self.params
        self.clock = 0.0                 # pipeline time (s): sum of step dt
        self.version = 0
        self._presence_off = 0.0
        self._ink_key = None
        self._ink = None
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
        self.presence.reset()
        apply_params(self.effective, self.tree, self.tracker)

    def _mix(self) -> dict:
        return idle_mix(self.params, self.scene.looks, self.idle.value())

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
            else:
                self.sources[cam_id].reader.update(cam.source)

    def set_scene(self, scene: Scene) -> None:
        """Adopt an edited scene: only what changed is rebuilt / reopened."""
        old = self.scene
        self.scene = scene.copy()
        new_params = P.normalise(self.scene.params)
        stage_changed = old.stage != self.scene.stage or new_params["proc_size"] != self.params["proc_size"]
        self.params = new_params
        self.effective = self._mix()
        if stage_changed:
            self._rebuild_stage()
        else:
            apply_params(self.effective, self.tree, self.tracker)
            old_cams = {c.id: c for c in old.cameras}
            moved = False
            for cam in self.scene.cameras:
                o = old_cams.get(cam.id)
                if o is None or o.transform != cam.transform or o.feather != cam.feather:
                    self.compositor.remove(cam.id)
                    moved = True
                elif o.enabled != cam.enabled or o.source != cam.source or o.color != cam.color:
                    moved = True
            for cam_id in set(old_cams) - {c.id for c in self.scene.cameras}:
                self.compositor.remove(cam_id)
                moved = True
            if moved:
                self.presence.relearn()          # the stage picture changed, not people
                self.glow.forget_environment()
        self._sync_sources()

    def set_params(self, values: dict) -> None:
        values = P.normalise(values)
        proc_changed = values["proc_size"] != self.params["proc_size"]
        self.params = values
        self.scene.params = dict(values)
        self.effective = self._mix()
        if proc_changed:
            self._rebuild_stage()
        else:
            apply_params(self.effective, self.tree, self.tracker)

    def relearn_background(self) -> None:
        """Learn the empty stage again from the next frames (nobody should be on it)."""
        self.presence.relearn()

    def _edge_ink(self, p: dict) -> tuple[bool, bool] | None:
        """Which edges of the ornament tile carry ink (for seamless colours), cached per tile."""
        key = (self.scene.resolve(p["tile"]), p["tile_crop"])
        if key != self._ink_key:
            self._ink_key = key
            try:
                self._ink = tile_edge_ink(load_tile(*key))
            except Exception as exc:
                log.warning("seamless colours: %s", exc)
                self._ink = None
            if self._ink is None:
                log.info("seamless colours: the tile has ink on all its edges or none; nothing to do")
        return self._ink

    # -- per frame -----------------------------------------------------------

    def step(self, dt: float) -> FrameResult:
        t0 = time.perf_counter()
        self.clock += dt
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

        p = self.effective
        cams = self.scene.cameras
        t = time.perf_counter()
        if p["mode"] == "depth":
            depth, valid, covered = self.compositor.compose_depth(cams, frames)
            raw = np.repeat(np.rint(depth).astype(np.uint8)[:, :, None], 3, axis=2)
        else:
            raw, covered = self.compositor.compose(cams, frames)
        timings["stitch"] = (time.perf_counter() - t) * 1000

        # People: motion against the learned empty stage (on the unfiltered picture,
        # so look changes to the input filters never read as movement)
        pres = None
        if needs_presence(p, self.idle_enabled):
            if self._presence_off > PRESENCE_STALE_SECONDS:
                self.presence.relearn()
            self._presence_off = 0.0
            t = time.perf_counter()
            pres = self.presence.update(raw, covered, dt, p["presence_threshold"], p["presence_absorb"],
                                        p["presence_trail"], p["presence_min_size"], p["proc_size"])
            timings["presence"] = (time.perf_counter() - t) * 1000
        else:
            self._presence_off += dt

        # Motion glow: from the depth picture, or the cameras' brightness in colour mode
        glow = None
        if p["glow"]:
            t = time.perf_counter()
            depth_mode = p["mode"] == "depth"
            glow = self.glow.update(depth if depth_mode else raw, valid if depth_mode else covered, dt,
                                    source=p["glow_from"], depth=depth_mode, threshold=p["glow_threshold"],
                                    memory=p["glow_memory"], flow=p["glow_flow"], fade=p["glow_fade"],
                                    cell_px=p["proc_size"])
            timings["glow"] = (time.perf_counter() - t) * 1000
        else:
            self.glow.clear()

        if p["mode"] == "depth":
            labels = self.layers.update(depth, valid, p["depth_min"], p["depth_max"], p["depth_layers"])
            palette = depth_palette(p["depth_layers"], p["depth_palette"], hex_to_bgr(p["depth_fg"]),
                                    hex_to_bgr(p["depth_bg"]), glow=p["glow"])
            stage_input = raw
        else:
            t = time.perf_counter()
            img = apply_blur(raw, p["blur"])
            img = apply_sharpen(img, p["sharpen"])
            stage_input = apply_filters(img, p["brightness"], p["contrast"], p["saturation"])
            timings["filter"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            fit = covered
            if pres is not None and p["presence"] and p["presence_palette"]:
                # Colours from the people, so clothes keep their colours (the rest is faded anyway)
                people = covered & (pres.activity > 0.3)
                if people.sum() >= PALETTE_MIN_PEOPLE * people.size:
                    fit = people
            labels, palette = self.tracker.update(stage_input, dt, mask=fit)
            # Uncovered pixels get an extra label drawn in the background colour (no ink)
            k = len(palette)
            labels = np.where(covered, labels, k)
            palette = np.vstack([palette, np.array(hex_to_bgr(p["background"]), np.float32)[None]])
            timings["posterize"] = (time.perf_counter() - t) * 1000

        if glow is not None:
            labels, palette = glow_labels(labels, palette, glow, glow_colour(p), render_background(p))

        t = time.perf_counter()
        self.tree.edge_ink = self._edge_ink(p) if p["seamless"] else None
        if pres is not None and p["presence"]:
            activity = pres.activity if glow is None else np.maximum(pres.activity, glow)
            self.tree.update(labels, palette, dt, activity=activity, outline=pres.mask)
        else:
            self.tree.update(labels, palette, dt)
        timings["tree"] = (time.perf_counter() - t) * 1000

        waves = None
        if pres is not None and p["waves"]:
            waves = self.waves.update(pres.people, self.clock, p["wave_interval"], p["wave_origin"],
                                      p["wave_range"] / p["wave_speed"])
        else:
            self.waves.clear()

        # Idle look: fade in after nobody for a while, out when someone comes (or when switched off)
        use_idle = self.idle_enabled and pres is not None and p["idle_look"] in self.scene.looks
        occupied = not use_idle or bool(pres.people) or pres.learning
        self.idle.update(occupied, dt, p["idle_after"], p["idle_fade"])
        eff = self._mix()
        if eff != self.effective:            # takes effect from the next frame
            self.effective = eff
            apply_params(eff, self.tree, self.tracker)

        self.version += 1
        snap = self.tree.snapshot()
        snap.version = self.version
        snap.waves = waves
        snap.clock = self.clock
        snap.wall = time.perf_counter()
        labels_image = palette[np.minimum(labels, len(palette) - 1)].astype(np.uint8)
        timings["total"] = (time.perf_counter() - t0) * 1000
        thumbs = {cid: img for cid, img in self.compositor.last_small.items() if cid in frames}
        sizes = {cid: (f.shape[1], f.shape[0]) for cid, f in frames.items()}
        return FrameResult(snap, stage_input, labels_image, covered, timings, status,
                           leaves=self.tree.leaf_count, thumbnails=thumbs, frame_sizes=sizes,
                           activity=None if pres is None else pres.activity,
                           people=[] if pres is None else pres.people,
                           learning=bool(pres is not None and pres.learning), idle=self.idle.value())

    def close(self) -> None:
        for src in self.sources.values():
            src.stop()
        self.sources.clear()


class PipelineRunner:
    """Runs a Pipeline continuously on a background thread.

    latest() returns the newest FrameResult (or None); changes are queued and
    applied between frames, so the caller never blocks on processing.
    """

    def __init__(self, scene: Scene, max_fps: float = 60.0, realtime: bool = True, idle: bool = True) -> None:
        self.max_fps = max_fps
        self.realtime = realtime
        self.idle = idle
        self._scene = scene
        self._latest: FrameResult | None = None
        self._lock = threading.Lock()
        self._commands: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.errors = 0
        self.last_error = ""
        self.fps = 0.0
        self.paused = False
        self.pipeline: Pipeline | None = None
        self.inject_fault = False          # testing: make the next step raise
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

    def relearn_background(self) -> None:
        """Learn the empty stage again (nobody should be on it for the next half second)."""
        self._commands.put(("relearn", None))

    def set_idle(self, on: bool) -> None:
        """Allow the idle look (perform.py: on; the editor: only when asked)."""
        self.idle = bool(on)
        self._commands.put(("idle", bool(on)))

    def _apply_commands(self, pipe: Pipeline) -> None:
        # Coalesce: only the newest scene / params matter
        scene = params = None
        relearn = False
        while True:
            try:
                kind, value = self._commands.get_nowait()
            except queue.Empty:
                break
            if kind == "scene":
                scene, params = value, None
            elif kind == "params":
                params = value
            elif kind == "relearn":
                relearn = True
            elif kind == "idle":
                pipe.idle_enabled = value
        if scene is not None:
            pipe.set_scene(scene)
        if params is not None:
            pipe.set_params(params)
        if relearn:
            pipe.relearn_background()

    def _run(self) -> None:
        try:
            pipe = Pipeline(self._scene, realtime=self.realtime)
            pipe.idle_enabled = self.idle
            self.pipeline = pipe
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
                    if self.inject_fault:
                        self.inject_fault = False
                        raise RuntimeError("injected fault")
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
