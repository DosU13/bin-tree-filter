"""Scene model: stage, cameras, projectors, look parameters; saved as JSON.

A scene file is what Align Perform edits and perform.py plays.  Positions and
sizes are in world pixels.  Unknown fields are kept so newer files survive a
round trip through older code; `version` drives migrations.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from core import geometry as geo
from core import params as P

SCENE_VERSION = 1
SOURCE_KINDS = ("webcam", "video", "image", "gif")
OUTPUT_KINDS = ("screen", "window", "none")
BACKENDS = ("any", "msmf", "dshow")
ROTATIONS = (0, 90, 180, 270)


class SceneError(ValueError):
    """Invalid scene file."""


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """Where the ornament lives: cols x rows square cells of cell_size world px."""

    x: float = 0.0
    y: float = 0.0
    cols: int = 1
    rows: int = 1
    cell_size: float = 1080.0
    levels: int = 12        # deepest bintree level (tree cost doubles per level)

    @property
    def width(self) -> float:
        return self.cols * self.cell_size

    @property
    def height(self) -> float:
        return self.rows * self.cell_size

    def world_to_cell(self) -> np.ndarray:
        s = 1.0 / self.cell_size
        return np.array([[s, 0.0, -self.x * s], [0.0, s, -self.y * s], [0.0, 0.0, 1.0]])

    def cell_to_world(self) -> np.ndarray:
        return np.array([[self.cell_size, 0.0, self.x], [0.0, self.cell_size, self.y], [0.0, 0.0, 1.0]])

    def quad(self) -> np.ndarray:
        return np.array([[self.x, self.y], [self.x + self.width, self.y],
                         [self.x + self.width, self.y + self.height], [self.x, self.y + self.height]])


@dataclass
class Transform:
    """Placement of an image (camera view or projector output) in the world.

    Without `corners` the image fills the rectangle (x, y, width, height);
    with corners (TL, TR, BR, BL world points) it is warped onto that quad.
    `rotation` / `mirror` apply to camera frames before placement.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 1920.0
    height: float = 1080.0
    rotation: int = 0
    mirror: bool = False
    corners: list[list[float]] | None = None

    def quad(self) -> np.ndarray:
        if self.corners is not None:
            return np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        x, y, w, h = self.x, self.y, self.width, self.height
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)

    def norm_to_world(self) -> np.ndarray:
        """Normalised display coords -> world."""
        if self.corners is not None:
            return geo.quad_matrix(self.corners)
        return geo.rect_matrix(self.x, self.y, self.width, self.height)

    def bounds(self) -> tuple[float, float, float, float]:
        q = self.quad()
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    def set_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.x, self.y, self.width, self.height = x, y, w, h
        self.corners = None


@dataclass
class SourceSpec:
    """Where a camera's frames come from.

    width / height: for webcams the requested capture size (0 = driver default);
    after probing, the real frame size.  For files, the probed frame size.
    """

    kind: str = "webcam"
    index: int = 0
    device_name: str = ""
    backend: str = "any"
    path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    noise: float = 0.0       # image sources: simulated sensor noise (std, 0..255)

    def label(self) -> str:
        if self.kind == "webcam":
            return f"Webcam {self.index}" + (f" ({self.device_name})" if self.device_name else "")
        return os.path.basename(self.path) or self.kind


@dataclass
class ColorFix:
    """Per-camera colour correction, applied before stitching."""

    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0


@dataclass
class CameraSpec:
    id: str
    name: str = ""
    enabled: bool = True
    source: SourceSpec = field(default_factory=SourceSpec)
    transform: Transform = field(default_factory=Transform)
    color: ColorFix = field(default_factory=ColorFix)
    feather: float = 100.0      # world px over which overlapping cameras cross-fade

    def frame_size(self) -> tuple[int, int]:
        """Size of the camera's frames as known so far (probed), else 16:9."""
        s = self.source
        if s.width > 0 and s.height > 0:
            return s.width, s.height
        return 1920, 1080

    def display_aspect(self) -> float:
        w, h = geo.display_size(*self.frame_size(), self.transform.rotation)
        return w / h


@dataclass
class OutputSpec:
    """Where a projector's image goes.

    screen: full-screen on a display, matched by serial, then name, then index.
    window: a normal window (testing on one monitor).
    none:   not shown (still drawn in Align Perform's canvas).
    """

    kind: str = "window"
    screen_name: str = ""
    screen_serial: str = ""
    screen_index: int = -1
    width: int = 1920
    height: int = 1080


@dataclass
class BlendSettings:
    feather: float | None = None     # world px; None = auto (the overlap width)
    gamma: float = 2.2


@dataclass
class ProjectorSpec:
    id: str
    name: str = ""
    enabled: bool = True
    output: OutputSpec = field(default_factory=OutputSpec)
    transform: Transform = field(default_factory=Transform)
    blend: BlendSettings = field(default_factory=BlendSettings)
    gain: float = 1.0

    def aspect(self) -> float:
        return self.output.width / max(1, self.output.height)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    stage: Stage = field(default_factory=Stage)
    cameras: list[CameraSpec] = field(default_factory=list)
    projectors: list[ProjectorSpec] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=P.defaults)
    looks: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: int = SCENE_VERSION
    extra: dict[str, Any] = field(default_factory=dict)
    path: str | None = None          # file it was loaded from / saved to (not serialised)

    # -- lookup --------------------------------------------------------------

    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()

    def resolve(self, path: str) -> str:
        """Absolute path for a file referenced by the scene (relative to the scene file)."""
        if not path:
            return path
        if os.path.isabs(path):
            return path
        candidate = os.path.join(self.base_dir, path)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
        # Fall back to the project root (bundled tiles / sample media)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(root, path)
        return os.path.normpath(alt if os.path.exists(alt) else candidate)

    def camera(self, cam_id: str) -> CameraSpec:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(cam_id)

    def projector(self, proj_id: str) -> ProjectorSpec:
        for p in self.projectors:
            if p.id == proj_id:
                return p
        raise KeyError(proj_id)

    def item(self, kind: str, item_id: str):
        return self.camera(item_id) if kind == "camera" else self.projector(item_id)

    def new_id(self, prefix: str) -> str:
        used = {c.id for c in self.cameras} | {p.id for p in self.projectors}
        i = 1
        while f"{prefix}{i}" in used:
            i += 1
        return f"{prefix}{i}"

    def enabled_cameras(self) -> list[CameraSpec]:
        return [c for c in self.cameras if c.enabled]

    def enabled_projectors(self) -> list[ProjectorSpec]:
        return [p for p in self.projectors if p.enabled]

    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    # -- serialisation -------------------------------------------------------

    def to_dict(self, relative_to: str | None = None) -> dict[str, Any]:
        def rel(path: str) -> str:
            if not path or relative_to is None or not os.path.isabs(path):
                return path
            try:
                r = os.path.relpath(path, relative_to)
            except ValueError:          # other drive
                return path
            # Files next to / below the scene stay relative so the folder can move
            return path if r.startswith("..") else r.replace(os.sep, "/")

        cams = []
        for c in self.cameras:
            d = asdict(c)
            d["source"]["path"] = rel(c.source.path)
            cams.append(d)
        projs = []
        for p in self.projectors:
            d = asdict(p)
            d["blend"]["feather"] = "auto" if p.blend.feather is None else p.blend.feather
            projs.append(d)
        params = dict(self.params)
        if params.get("tile") and os.path.isabs(params["tile"]):
            params["tile"] = rel(params["tile"])
        out = dict(self.extra)
        out.update({
            "version": self.version,
            "stage": asdict(self.stage),
            "cameras": cams,
            "projectors": projs,
            "params": params,
            "looks": self.looks,
        })
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str | None = None) -> "Scene":
        if not isinstance(data, dict):
            raise SceneError("scene file must contain a JSON object")
        version = int(data.get("version", 1))
        if version > SCENE_VERSION:
            raise SceneError(f"scene version {version} is newer than this program ({SCENE_VERSION})")
        known = {"version", "stage", "cameras", "projectors", "params", "looks"}
        try:
            scene = cls(
                stage=_build(Stage, data.get("stage", {})),
                cameras=[_camera(c) for c in data.get("cameras", [])],
                projectors=[_projector(p) for p in data.get("projectors", [])],
                params=P.normalise(data.get("params", {})),
                looks={str(k): P.normalise(v) for k, v in (data.get("looks") or {}).items()},
                version=SCENE_VERSION,
                extra={k: v for k, v in data.items() if k not in known},
                path=path,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise SceneError(str(exc)) from exc
        scene.validate()
        return scene

    def validate(self) -> None:
        s = self.stage
        if s.cols < 1 or s.rows < 1 or s.cols * s.rows > 256:
            raise SceneError(f"stage must have 1..256 cells, got {s.cols} x {s.rows}")
        if not (s.cell_size > 0 and math.isfinite(s.cell_size)):
            raise SceneError("stage cell_size must be positive")
        if not 1 <= s.levels <= 16:
            raise SceneError("stage levels must be 1..16")
        ids = [c.id for c in self.cameras] + [p.id for p in self.projectors]
        if len(ids) != len(set(ids)):
            raise SceneError("camera / projector ids must be unique")
        for c in self.cameras:
            if c.source.kind not in SOURCE_KINDS:
                raise SceneError(f"camera {c.id}: unknown source kind {c.source.kind!r}")
            if c.source.backend not in BACKENDS:
                raise SceneError(f"camera {c.id}: unknown backend {c.source.backend!r}")
            if c.transform.rotation not in ROTATIONS:
                raise SceneError(f"camera {c.id}: rotation must be one of {ROTATIONS}")
            _check_transform(c.transform, f"camera {c.id}")
        for p in self.projectors:
            if p.output.kind not in OUTPUT_KINDS:
                raise SceneError(f"projector {p.id}: unknown output kind {p.output.kind!r}")
            if p.output.width < 1 or p.output.height < 1:
                raise SceneError(f"projector {p.id}: output size must be positive")
            _check_transform(p.transform, f"projector {p.id}")

    @classmethod
    def load(cls, path: str) -> "Scene":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise SceneError(f"{os.path.basename(path)}: not valid JSON ({exc})") from exc
        return cls.from_dict(data, path=os.path.abspath(path))

    def save(self, path: str | None = None) -> None:
        """Write JSON atomically (temp file + rename), paths relative to the file."""
        path = os.path.abspath(path or self.path)
        if not path:
            raise SceneError("no path to save to")
        self.validate()
        data = self.to_dict(relative_to=os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
        self.path = path


def _build(cls, data: dict[str, Any]):
    """Dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        raise SceneError(f"{cls.__name__} must be an object")
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def _transform(data: dict[str, Any]) -> Transform:
    t = _build(Transform, data)
    t.rotation = int(t.rotation)
    t.mirror = bool(t.mirror)
    if t.corners is not None:
        t.corners = [[float(a), float(b)] for a, b in t.corners]
        if len(t.corners) != 4:
            raise SceneError("transform corners must be 4 points")
    return t


def _check_transform(t: Transform, what: str) -> None:
    if t.corners is None and (t.width <= 0 or t.height <= 0):
        raise SceneError(f"{what}: width and height must be positive")
    if not np.all(np.isfinite(t.quad())):
        raise SceneError(f"{what}: position is not a number")


def _camera(d: dict[str, Any]) -> CameraSpec:
    if "id" not in d:
        raise SceneError("every camera needs an id")
    cam = _build(CameraSpec, {k: v for k, v in d.items() if k not in ("source", "transform", "color")})
    cam.id = str(cam.id)
    cam.source = _build(SourceSpec, d.get("source", {}))
    cam.transform = _transform(d.get("transform", {}))
    cam.color = _build(ColorFix, d.get("color", {}))
    return cam


def _projector(d: dict[str, Any]) -> ProjectorSpec:
    if "id" not in d:
        raise SceneError("every projector needs an id")
    proj = _build(ProjectorSpec, {k: v for k, v in d.items() if k not in ("output", "transform", "blend")})
    proj.id = str(proj.id)
    proj.output = _build(OutputSpec, d.get("output", {}))
    proj.transform = _transform(d.get("transform", {}))
    blend = dict(d.get("blend", {}))
    if blend.get("feather") in ("auto", None, ""):
        blend["feather"] = None
    proj.blend = _build(BlendSettings, blend)
    return proj
