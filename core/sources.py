"""Frame sources: webcam, video file, animated GIF, still image (all return BGR frames).

Two families:

* live.py's simple sources (CameraSource, VideoSource, GifSource, ImageSource,
  open_source): one call = one frame.
* Scene sources for Align Perform / perform.py.  A FrameReader knows how to open
  a device or file and read one frame.  LiveSource runs a reader on a
  background thread: latest() never blocks, files play at their own frame rate,
  and a failing camera is reopened with back-off while its last good frame stays
  available.  SyncSource reads the next frame on every call (deterministic; for
  tools and tests).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv")


# ---------------------------------------------------------------------------
# live.py sources
# ---------------------------------------------------------------------------

class CameraSource:
    """Webcam read on a background thread, so capture never blocks processing."""

    def __init__(self, index: int) -> None:
        print(f"Opening camera {index} ...", flush=True)
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened() and os.name == "nt":
            self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.is_opened = self._cap.isOpened()
        self._frame = None
        self._lock = threading.Lock()
        self._running = self.is_opened
        if self.is_opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def read(self, timeout: float = 5.0) -> tuple[bool, np.ndarray | None]:
        deadline = time.perf_counter() + timeout
        while self._running and self._frame is None and time.perf_counter() < deadline:
            time.sleep(0.005)
        if self._frame is None:
            print("Error: camera opened but delivers no frames "
                  "(in use by another app, unplugged, or disabled)")
        with self._lock:
            return self._frame is not None, self._frame

    def release(self) -> None:
        self._running = False
        self._cap.release()


class VideoSource:
    """Video file read frame by frame, looping at the end."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        self.is_opened = self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        self._cap.release()


class GifSource:
    """Animated GIF, looping."""

    def __init__(self, path: str) -> None:
        self._gif = Image.open(path)
        self._n = self._gif.n_frames
        self._idx = 0
        self.is_opened = True

    def read(self) -> tuple[bool, np.ndarray]:
        self._gif.seek(self._idx % self._n)
        self._idx += 1
        return True, cv2.cvtColor(np.array(self._gif.convert("RGB")), cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self._gif.close()


class ImageSource:
    """Still image with simulated sensor noise and 1-px jitter, like a webcam."""

    def __init__(self, path: str, noise: float = 4.0) -> None:
        self._img = cv2.imread(path)
        self.is_opened = self._img is not None
        self._noise = noise
        self._rng = np.random.default_rng()

    def read(self) -> tuple[bool, np.ndarray]:
        dx, dy = self._rng.integers(-1, 2, size=2)
        img = np.roll(self._img, (dy, dx), axis=(0, 1)).astype(np.float32)
        img += self._rng.normal(0, self._noise, img.shape)
        return True, np.clip(img, 0, 255).astype(np.uint8)

    def release(self) -> None:
        pass


def open_source(arg: str | None):
    """Webcam index ("0", "1", None) or path to a video, GIF or image."""
    if arg is None or arg.isdigit():
        return CameraSource(int(arg or 0))
    ext = os.path.splitext(arg)[1].lower()
    if ext == ".gif":
        return GifSource(arg)
    if ext in IMAGE_EXTS:
        return ImageSource(arg)
    return VideoSource(arg)


# ---------------------------------------------------------------------------
# Scene sources
# ---------------------------------------------------------------------------

BACKEND_CODES = {"any": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
STATUSES = ("starting", "ok", "stalled", "reconnecting", "stopped")


class SourceError(RuntimeError):
    """A source could not be opened or delivers no frames."""


@dataclass
class CameraInfo:
    index: int
    name: str
    path: str
    backend: int


def list_cameras() -> list[CameraInfo]:
    """Webcams attached to this machine (index usable with cv2.VideoCapture)."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return []
    backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_ANY
    try:
        cams = enumerate_cameras(backend)
    except Exception as exc:          # driver quirks must not break the UI
        log.warning("camera enumeration failed: %s", exc)
        return []
    return [CameraInfo(c.index, c.name, c.path, c.backend) for c in cams]


def resolve_camera_index(spec) -> int:
    """Index of the webcam a spec means: by device name when known (Windows may renumber)."""
    if spec.device_name:
        matches = [c.index for c in list_cameras() if c.name == spec.device_name]
        if matches:
            return spec.index if spec.index in matches else matches[0]
    return spec.index


def kind_for_path(path: str) -> str:
    """Source kind for a file path, from its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        return "gif"
    if ext in IMAGE_EXTS:
        return "image"
    return "video"


class FrameReader:
    """Opens one device / file and reads frames from it."""

    frame_interval = 0.0      # seconds between frames for files; 0 = device paces itself

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray | None:
        """Next BGR frame, or None on failure."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class WebcamReader(FrameReader):
    def __init__(self, spec) -> None:
        self.spec = spec
        self._cap = None

    def open(self) -> None:
        s = self.spec
        index = resolve_camera_index(s)
        code = BACKEND_CODES.get(s.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, code)
        if not cap.isOpened() and s.backend == "any" and os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SourceError(f"cannot open webcam {index}")
        if s.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc[:4].ljust(4)))
        if s.width > 0 and s.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        if s.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, s.fps)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoReader(FrameReader):
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._cap = None

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise SourceError(f"video not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise SourceError(f"cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / fps if 1 <= fps <= 240 else 1 / 30
        self._cap = cap

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class GifReader(FrameReader):
    def __init__(self, path: str) -> None:
        self.path = path
        self._gif = None
        self._idx = 0
        self._n = 1

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise SourceError(f"GIF not found: {self.path}")
        self._gif = Image.open(self.path)
        self._n = getattr(self._gif, "n_frames", 1)
        self._idx = 0
        self.frame_interval = max(0.01, self._gif.info.get("duration", 40) / 1000)

    def read(self) -> np.ndarray | None:
        self._gif.seek(self._idx % self._n)
        self._idx += 1
        return cv2.cvtColor(np.array(self._gif.convert("RGB")), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._gif is not None:
            self._gif.close()
            self._gif = None


class ImageReader(FrameReader):
    """Still image; with noise > 0, sensor noise and 1-px jitter like a webcam."""

    frame_interval = 1 / 30

    def __init__(self, path: str, noise: float = 0.0) -> None:
        self.path = path
        self.noise = noise
        self._img = None
        self._rng = np.random.default_rng()

    def open(self) -> None:
        img = cv2.imread(self.path)
        if img is None:
            raise SourceError(f"cannot read image: {self.path}")
        self._img = img

    def read(self) -> np.ndarray | None:
        if self.noise <= 0:
            return self._img
        dx, dy = self._rng.integers(-1, 2, size=2)
        img = np.roll(self._img, (dy, dx), axis=(0, 1)).astype(np.float32)
        img += self._rng.normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def make_reader(spec, resolve=lambda p: p) -> FrameReader:
    """Reader for a core.scene.SourceSpec; *resolve* maps scene paths to absolute paths."""
    if spec.kind == "webcam":
        return WebcamReader(spec)
    path = resolve(spec.path)
    if spec.kind == "video":
        return VideoReader(path)
    if spec.kind == "gif":
        return GifReader(path)
    if spec.kind == "image":
        return ImageReader(path, spec.noise)
    raise SourceError(f"unknown source kind {spec.kind!r}")


def probe(spec, resolve=lambda p: p, timeout: float = 10.0) -> tuple[int, int, float]:
    """Open a source briefly and return its real (width, height, fps)."""
    reader = make_reader(spec, resolve)
    reader.open()
    try:
        deadline = time.perf_counter() + timeout
        frame = reader.read()
        while frame is None and time.perf_counter() < deadline:
            time.sleep(0.05)
            frame = reader.read()
        if frame is None:
            raise SourceError(f"{spec.label()}: opened but delivers no frames")
        fps = 1.0 / reader.frame_interval if reader.frame_interval > 0 else float(spec.fps or 30)
        return frame.shape[1], frame.shape[0], round(fps, 2)
    finally:
        reader.close()


class LiveSource:
    """Background-thread source with pacing, status and automatic reconnect."""

    def __init__(self, reader: FrameReader, name: str = "", stall_timeout: float = 3.0,
                 max_backoff: float = 30.0) -> None:
        self.reader = reader
        self.name = name
        self.stall_timeout = stall_timeout
        self.max_backoff = max_backoff
        self.error = ""
        self.reconnects = 0
        self._status = "starting"
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._last_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"source-{name}", daemon=True)

    def start(self) -> "LiveSource":
        self._thread.start()
        return self

    @property
    def status(self) -> str:
        if self._status == "ok" and self.age() > self.stall_timeout:
            return "stalled"
        return self._status

    def age(self) -> float:
        """Seconds since the last frame (inf before the first)."""
        return time.perf_counter() - self._last_time if self._last_time else float("inf")

    def latest(self) -> tuple[np.ndarray | None, int]:
        """Newest frame and its sequence number (frame may be None before the first)."""
        with self._lock:
            return self._frame, self._frame_id

    def _wait(self, seconds: float) -> bool:
        """Sleep unless stopped; True if stop was requested."""
        return self._stop.wait(max(0.0, seconds))

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.reader.open()
            except Exception as exc:
                self.error = str(exc)
                self._status = "reconnecting"
                log.warning("source %s: %s (retry in %.0f s)", self.name, exc, backoff)
                if self._wait(backoff):
                    break
                backoff = min(backoff * 2, self.max_backoff)
                continue
            backoff = 1.0
            failures = 0
            next_t = time.perf_counter()
            while not self._stop.is_set():
                try:
                    frame = self.reader.read()
                except Exception as exc:
                    self.error = str(exc)
                    frame = None
                if frame is None:
                    failures += 1
                    if failures >= 5:
                        break
                    if self._wait(0.05):
                        break
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                self._last_time = time.perf_counter()
                self._status = "ok"
                self.error = ""
                interval = self.reader.frame_interval
                if interval > 0:
                    next_t += interval
                    now = time.perf_counter()
                    if next_t < now - 1.0:          # fell far behind: resync
                        next_t = now
                    if self._wait(next_t - now):
                        break
            try:
                self.reader.close()
            except Exception:
                pass
            if not self._stop.is_set():
                self.reconnects += 1
                self._status = "reconnecting"
                log.warning("source %s lost (%s); reconnecting", self.name, self.error or "no frames")
                if self._wait(backoff):
                    break
        self._status = "stopped"

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)


class SyncSource:
    """Reads the next frame on every latest() call; same interface as LiveSource."""

    def __init__(self, reader: FrameReader, name: str = "") -> None:
        self.reader = reader
        self.name = name
        self.error = ""
        self.reconnects = 0
        self._frame_id = 0
        self._frame = None
        try:
            reader.open()
            self.status = "ok"
        except Exception as exc:
            self.error = str(exc)
            self.status = "reconnecting"

    def start(self) -> "SyncSource":
        return self

    def age(self) -> float:
        return 0.0 if self._frame is not None else float("inf")

    def latest(self) -> tuple[np.ndarray | None, int]:
        if self.status == "ok":
            frame = self.reader.read()
            if frame is not None:
                self._frame = frame
                self._frame_id += 1
        return self._frame, self._frame_id

    def stop(self, timeout: float = 0.0) -> None:
        self.reader.close()
        self.status = "stopped"


def open_scene_source(spec, resolve=lambda p: p, name: str = "", realtime: bool = True):
    """LiveSource (threaded, paced) or SyncSource (one frame per call) for a SourceSpec."""
    reader = make_reader(spec, resolve)
    if realtime:
        return LiveSource(reader, name=name).start()
    return SyncSource(reader, name=name)
