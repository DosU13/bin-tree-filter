"""Frame sources: webcam, video file, animated GIF, still image (all return BGR frames)."""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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
