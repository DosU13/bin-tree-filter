"""Orbbec Astra depth cameras (Astra, Astra S, Astra Mini) through OpenNI2.

An Astra is not a webcam: its depth and colour streams come through OpenNI2 with
Orbbec's driver.  Both ship in the Astra SDK's `bin` folder (OpenNI2.dll finds
orbbec.dll next to itself); the `openni` package loads them.  The runtime folder
is the first of these that holds OpenNI2.dll:

    1. $OPENNI2_REDIST64 / $OPENNI2_REDIST
    2. vendor/openni2 in the project (copy the SDK's bin folder there)
    3. an Astra SDK unpacked next to the project folder (../AstraSDK*/bin)

Depth frames are millimetres (0 = no reading).  AstraReader turns them into a
grey picture for depth mode: depth_near is white (255), depth_far and beyond
almost black (1), no reading black (0), so depth mode's band picks a slice of
the room.

Hardware notes (Astra S): plug it straight into the PC (on a shared USB hub
opening fails with "USB transfer timeout"); anything a few cm in front of it
trips its eye-safety cutoff, the laser goes off and every depth pixel reads 0;
it measures from about 0.4 m.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from core.sources import FrameReader, SourceError

log = logging.getLogger(__name__)

STREAMS = ("depth", "colour")
DEPTH_SIZES = ((640, 480), (320, 240), (160, 120), (1280, 1024))
COLOUR_SIZES = ((640, 480), (320, 240), (1280, 960))
READ_TIMEOUT = 0.5               # s without a frame before read() gives up (LiveSource reconnects after 5)
DARK_WARNING_FRAMES = 60         # all-zero depth frames in a row before warning about the laser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP_HINT = ("set OPENNI2_REDIST64 to the Astra SDK's bin folder, copy that folder to vendor/openni2, "
              "or unpack the Astra SDK next to the project folder")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def runtime_dir(env=None, root: str = ROOT) -> str | None:
    """Folder holding OpenNI2.dll and the Orbbec driver, or None if there is none."""
    env = os.environ if env is None else env
    candidates = [env.get("OPENNI2_REDIST64", ""), env.get("OPENNI2_REDIST", ""),
                  os.path.join(root, "vendor", "openni2")]
    parent = os.path.dirname(root)
    for pattern in ("AstraSDK*/bin", "AstraSDK*/AstraSDK*/bin"):
        candidates += sorted(glob.glob(os.path.join(parent, pattern)), reverse=True)    # newest version first
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "OpenNI2.dll" if os.name == "nt" else "libOpenNI2.so")):
            return os.path.normpath(d)
    return None


_lock = threading.RLock()
_ni = None                                   # the openni2 module once OpenNI2 is loaded
_devices: dict[bytes, list] = {}             # uri -> [Device, users]
_serials: dict[bytes, str] = {}              # uri -> serial number (learning one means opening the device)


def _message(exc: Exception) -> str:
    """Readable text for an OpenNIError (whose args are code, bytes message, log file)."""
    msg = exc.args[1] if len(exc.args) > 1 else str(exc)
    if isinstance(msg, bytes):
        msg = msg.decode(errors="replace")
    msg = str(msg).strip()
    if "timeout" in msg.lower():
        msg += " (plug the Astra straight into the PC, not into a hub)"
    return msg


def _openni():
    """The openni2 module with OpenNI2 loaded (once per process)."""
    global _ni
    if _ni is not None:
        return _ni
    with _lock:
        if _ni is not None:
            return _ni
        try:
            from openni import openni2
        except ImportError as exc:
            raise SourceError("the 'openni' package is missing (pip install -r requirements.txt)") from exc
        path = runtime_dir()
        if path is None:
            raise SourceError(f"OpenNI2 not found: {SETUP_HINT}")
        try:
            openni2.initialize(path)       # changes the working directory for a moment
        except Exception as exc:
            raise SourceError(f"cannot load OpenNI2 from {path}: {exc}") from exc
        log.info("OpenNI2 %s loaded from %s", openni2.get_version(), path)
        _ni = openni2
        return _ni


def _uris() -> list[bytes]:
    return sorted(_openni().Device.enumerate_uris())


def _ref(uri: bytes):
    """Open a device (or share the open one); every _ref needs a _release."""
    entry = _devices.get(uri)
    if entry is None:
        ni = _openni()
        try:
            dev = ni.Device(uri)
        except Exception as exc:
            raise SourceError(f"cannot open Astra: {_message(exc)}") from exc
        try:
            serial = dev.get_property(ni.c_api.ONI_DEVICE_PROPERTY_SERIAL_NUMBER, ctypes.c_char * 64).value
            _serials[uri] = serial.decode(errors="replace")
        except Exception:
            _serials.setdefault(uri, "")
        entry = _devices[uri] = [dev, 0]
    entry[1] += 1
    return entry[0]


def _release(uri: bytes) -> None:
    with _lock:
        entry = _devices.get(uri)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _devices[uri]
            try:
                entry[0].close()
            except Exception:
                pass


def _acquire(spec) -> tuple[bytes, object]:
    """The device a spec means: by serial (device_name) when it is connected, else by index."""
    with _lock:
        uris = _uris()
        if not uris:
            raise SourceError("no Astra connected")
        if spec.device_name:
            # Devices whose serial is known first; the others must be opened to find out
            for uri in sorted(uris, key=lambda u: _serials.get(u) != spec.device_name):
                if _serials.get(uri, spec.device_name) != spec.device_name:
                    continue
                try:
                    dev = _ref(uri)
                except SourceError:
                    continue
                if _serials.get(uri) == spec.device_name:
                    return uri, dev
                _release(uri)
        if not 0 <= spec.index < len(uris):
            raise SourceError(f"Astra {spec.index} is not connected ({len(uris)} found)")
        uri = uris[spec.index]
        return uri, _ref(uri)


@dataclass
class AstraInfo:
    index: int
    uri: bytes
    serial: str          # "" until the device has been opened once in this process

    def label(self) -> str:
        return f"Astra {self.index}" + (f" (serial {self.serial})" if self.serial else "")


def list_astras() -> list[AstraInfo]:
    """Connected Astras (cheap: no device is opened).  Raises SourceError without OpenNI2.

    Doesn't wait for _lock, which is held while a device opens (seconds)."""
    return [AstraInfo(i, uri, _serials.get(uri, "")) for i, uri in enumerate(_uris())]


def known_serial(uri: bytes) -> str:
    """Serial number of a device opened earlier in this process, else "" (never blocks)."""
    return _serials.get(uri, "")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def depth_lut(near_m: float, far_m: float) -> np.ndarray:
    """uint8 lookup table, millimetres -> grey: near 255 .. far 1, 0 (no reading) stays 0."""
    near, far = near_m * 1000.0, max(far_m * 1000.0, near_m * 1000.0 + 1.0)
    mm = np.arange(65536, dtype=np.float32)
    lut = np.clip(np.rint(255.0 - (mm - near) * (254.0 / (far - near))), 1, 255).astype(np.uint8)
    lut[0] = 0
    return lut


def depth_to_gray(depth_mm: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """Depth in millimetres (uint16) -> uint8 grey, see depth_lut."""
    return depth_lut(near_m, far_m)[depth_mm]


def _pick_mode(stream, size: tuple[int, int], fps: float, pixel_format):
    modes = [m for m in stream.get_sensor_info().videoModes if m.pixelFormat == pixel_format]
    fits = [m for m in modes if (m.resolutionX, m.resolutionY) == size]
    if not fits:
        sizes = sorted({(m.resolutionX, m.resolutionY) for m in modes})
        raise SourceError(f"Astra has no {size[0]}x{size[1]} mode (has {', '.join(f'{w}x{h}' for w, h in sizes)})")
    return min(fits, key=lambda m: (abs(m.fps - fps) if fps > 0 else -m.fps))


class AstraReader(FrameReader):
    """One stream (depth or colour) of an Astra; BGR frames."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.serial = ""
        self._uri: bytes | None = None
        self._stream = None
        self._lut_key = None
        self._lut = None
        self._dark = 0

    def open(self) -> None:
        s = self.spec
        if s.stream not in STREAMS:
            raise SourceError(f"unknown Astra stream {s.stream!r}")
        ni = _openni()
        uri, dev = _acquire(s)
        depth = s.stream == "depth"
        try:
            stream = dev.create_depth_stream() if depth else dev.create_color_stream()
            if stream is None:
                raise SourceError(f"this Astra has no {s.stream} stream")
            size = (s.width, s.height) if s.width > 0 and s.height > 0 else (640, 480)
            fmt = ni.PIXEL_FORMAT_DEPTH_1_MM if depth else ni.PIXEL_FORMAT_RGB888
            stream.set_video_mode(_pick_mode(stream, size, s.fps, fmt))
            stream.set_mirroring_enabled(False)          # like webcams: mirror in the scene if wanted
            stream.start()
        except SourceError:
            _release(uri)
            raise
        except Exception as exc:
            _release(uri)
            raise SourceError(f"Astra {s.stream}: {_message(exc)}") from exc
        self._uri, self._stream, self._dark = uri, stream, 0
        self.serial = _serials.get(uri, "")

    def update(self, spec) -> None:
        self.spec = spec                  # depth_near / depth_far apply from the next frame

    def _gray(self, mm: np.ndarray) -> np.ndarray:
        key = (self.spec.depth_near, self.spec.depth_far)
        if key != self._lut_key:
            self._lut_key, self._lut = key, depth_lut(*key)
        return self._lut[mm]

    def read(self) -> np.ndarray | None:
        ni = _openni()
        if ni.wait_for_any_stream([self._stream], READ_TIMEOUT) is None:
            return None
        f = self._stream.read_frame()
        if self.spec.stream == "depth":
            mm = np.frombuffer(f.get_buffer_as_uint16(), np.uint16).reshape(f.height, f.width)
            gray = self._gray(mm)
            self._check_dark(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = np.frombuffer(f.get_buffer_as_uint8(), np.uint8).reshape(f.height, f.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _check_dark(self, gray: np.ndarray) -> None:
        if gray.any():
            if self._dark >= DARK_WARNING_FRAMES:
                log.info("Astra %s: depth readings are back", self.serial or self.spec.index)
            self._dark = 0
            return
        self._dark += 1
        if self._dark == DARK_WARNING_FRAMES:
            log.warning("Astra %s: no depth readings for 2 s; something within a few cm of the camera "
                        "switches its laser off (or everything is closer than 0.4 m)", self.serial or self.spec.index)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            for call in (stream.stop, stream.close):
                try:
                    call()
                except Exception:
                    pass
        if self._uri is not None:
            _release(self._uri)
            self._uri = None
