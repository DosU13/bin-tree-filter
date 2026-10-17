"""Every tunable option, declared once.

The Align Perform look panel, its hotkeys, scene files and look crossfades are
all generated from PARAMS, so adding an option is one new Param(...) line plus
the code that reads it (core.pipeline.apply_params).

Values live in plain dicts {key: value} so they serialise to JSON as-is.
Colours are '#rrggbb' strings; convert with core.filters.hex_to_bgr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.filters import BG_PRESETS, FG_PRESETS, bgr_to_hex, hex_to_bgr
from core.posterize import PALETTE_STYLES

KINDS = ("int", "float", "bool", "choice", "color", "path")
MODES = ("colour", "depth")


@dataclass(frozen=True)
class Param:
    """One option.

    Attributes:
        key:      Name used in dicts, JSON and code.
        label:    Text shown in the UI.
        group:    UI section.
        kind:     One of KINDS.
        default:  Default value.
        min, max: Range for numbers.
        step:     Hotkey / spin-box increment for numbers.
        choices:  Options for "choice"; presets cycled by the hotkey for "color".
        keys:     Hotkeys: (decrease, increase) for numbers, (toggle,) / (cycle,) otherwise.
        mode:     Only relevant in this mode ("colour" / "depth"); None = always.
        help:     Tooltip.
    """

    key: str
    label: str
    group: str
    kind: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = ()
    keys: tuple[str, ...] = ()
    mode: str | None = None
    help: str = ""

    @property
    def decimals(self) -> int:
        if self.kind != "float" or self.step is None:
            return 0
        text = f"{self.step:.6f}".rstrip("0")
        return max(1, len(text.split(".")[1]) if "." in text else 0)


PARAMS: tuple[Param, ...] = (
    # --- Mode --------------------------------------------------------------
    Param("mode", "Mode", "Mode", "choice", "colour", choices=MODES,
          help="colour: posterize the cameras; depth: two-colour depth band"),

    # --- Tree --------------------------------------------------------------
    Param("max_depth", "Max depth", "Tree", "int", 8, 1, 12, 1, keys=("d", "D"),
          help="Deepest split level (smaller triangles)"),
    Param("min_depth", "Min depth", "Tree", "int", 0, 0, 11, 1, keys=("m", "M"),
          help="Always split at least this deep"),
    Param("threshold", "Split threshold", "Tree", "float", 0.75, 0.01, 1.0, 0.01, keys=("h", "H"),
          help="Split a triangle when its main colour covers less than this share"),
    Param("hysteresis", "Merge hysteresis", "Tree", "float", 0.08, 0.0, 0.5, 0.01,
          help="Merge only when purity is this much above the split threshold"),
    Param("color_hysteresis", "Colour hysteresis", "Tree", "float", 0.15, 0.0, 0.5, 0.01,
          help="Keep a triangle's colour unless another beats it by this share"),
    Param("conforming", "Conforming mesh", "Tree", "bool", True, keys=("v",),
          help="Neighbouring triangles differ by at most one level (no T-junctions)"),
    Param("animate", "Animate", "Tree", "bool", True, keys=("i",),
          help="Cross-fade splits and colours instead of switching instantly"),
    Param("split_time", "Split time (s)", "Tree", "float", 0.15, 0.0, 2.0, 0.05,
          help="Duration of one level's cross-fade"),
    Param("tau_hist", "Statistics smoothing (s)", "Tree", "float", 0.15, 0.0, 2.0, 0.05,
          help="Time constant of the per-triangle colour statistics"),
    Param("tau_color", "Colour smoothing (s)", "Tree", "float", 0.12, 0.0, 2.0, 0.05,
          help="Time constant of triangle colour changes"),

    # --- Colour ------------------------------------------------------------
    Param("colors", "Colours", "Colour", "int", 4, 2, 8, 1, keys=("c", "C"), mode="colour",
          help="Number of posterize colours"),
    Param("palette_style", "Palette style", "Colour", "choice", "natural",
          choices=tuple(PALETTE_STYLES), keys=("k",), mode="colour"),
    Param("palette_tau", "Palette smoothing (s)", "Colour", "float", 0.6, 0.0, 5.0, 0.1, mode="colour",
          help="How fast the palette follows the scene"),
    Param("label_margin", "Label margin", "Colour", "float", 4.0, 0.0, 20.0, 0.5, mode="colour",
          help="Lab units a pixel must gain before it switches colour (anti-flicker)"),

    # --- Input -------------------------------------------------------------
    Param("blur", "Blur", "Input", "int", 0, 0, 20, 1, keys=("[", "]")),
    Param("sharpen", "Sharpen", "Input", "float", 0.0, 0.0, 5.0, 0.2, keys=(";", "'")),
    Param("brightness", "Brightness", "Input", "int", 0, -100, 100, 10, keys=("b", "B"), mode="colour"),
    Param("contrast", "Contrast", "Input", "float", 1.0, 0.1, 3.0, 0.1, keys=("n", "N")),
    Param("saturation", "Saturation", "Input", "float", 1.0, 0.0, 3.0, 0.1, keys=("t", "T")),

    # --- Depth -------------------------------------------------------------
    Param("depth_min", "Band min", "Depth", "int", 60, 0, 250, 5, keys=("a", "A"), mode="depth"),
    Param("depth_max", "Band max", "Depth", "int", 140, 5, 255, 5, keys=("z", "Z"), mode="depth"),
    Param("depth_fg", "Foreground", "Depth", "color", bgr_to_hex(FG_PRESETS[0]),
          choices=tuple(bgr_to_hex(c) for c in FG_PRESETS), keys=("f",), mode="depth"),
    Param("depth_bg", "Background", "Depth", "color", bgr_to_hex(BG_PRESETS[0]),
          choices=tuple(bgr_to_hex(c) for c in BG_PRESETS), keys=("b",), mode="depth"),

    # --- Render ------------------------------------------------------------
    Param("polarity", "Polarity", "Render", "choice", "alternate",
          choices=("alternate", "none", "invert"), keys=("x",),
          help="Which levels draw the ornament inverted"),
    Param("cord", "Cord", "Render", "bool", False, keys=("g",),
          help="Background-coloured seam between triangles of different colour"),
    Param("wireframe", "Wireframe", "Render", "bool", False, keys=("e",),
          help="Outline every triangle"),
    Param("background", "Background", "Render", "color", bgr_to_hex((181, 217, 253)), mode="colour",
          help="Colour behind the ornament ink"),
    Param("tile", "Ornament tile", "Render", "path", "ornam.jpg",
          help="Black-on-white ornament image"),
    Param("tile_crop", "Crop tile margin", "Render", "bool", False,
          help="Trim the tile's white margin so the ink touches the edges"),
    Param("proc_size", "Statistics resolution", "Render", "int", 128, 32, 512, 32, keys=("p", "P"),
          help="Pixels per stage cell used for colour statistics"),
)

BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}
GROUPS: tuple[str, ...] = tuple(dict.fromkeys(p.group for p in PARAMS))


def defaults() -> dict[str, Any]:
    return {p.key: p.default for p in PARAMS}


def coerce(key: str, value: Any) -> Any:
    """Convert *value* to the param's type and clamp it to its range."""
    p = BY_KEY[key]
    if p.kind == "int":
        v = int(round(float(value)))
    elif p.kind == "float":
        v = float(value)
    elif p.kind == "bool":
        v = value.lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
    elif p.kind == "choice":
        v = str(value)
        if v not in p.choices:
            raise ValueError(f"{key}: {v!r} is not one of {p.choices}")
        return v
    elif p.kind == "color":
        if isinstance(value, (tuple, list)):
            return bgr_to_hex(value)
        hex_to_bgr(str(value))   # validates
        return str(value).lower()
    else:
        return str(value)
    if p.min is not None:
        v = max(p.min, v)
    if p.max is not None:
        v = min(p.max, v)
    return type(v)(v)


def normalise(values: dict[str, Any]) -> dict[str, Any]:
    """Coerce every known value, fill missing ones with defaults, enforce cross-param rules.

    Unknown keys are dropped (older / newer scene files stay loadable).
    """
    out = defaults()
    for key, value in values.items():
        if key in BY_KEY:
            try:
                out[key] = coerce(key, value)
            except (TypeError, ValueError):
                pass
    out["min_depth"] = min(out["min_depth"], out["max_depth"] - 1)
    out["depth_min"] = min(out["depth_min"], out["depth_max"] - 5)
    return out


def nudge(values: dict[str, Any], key: str, direction: int) -> Any:
    """New value of *key* after one hotkey press (+1 / -1; direction is ignored for toggles)."""
    p = BY_KEY[key]
    v = values[key]
    if p.kind in ("int", "float"):
        new = v + direction * p.step
        if p.kind == "float":
            new = round(new, p.decimals)
        return coerce(key, new)
    if p.kind == "bool":
        return not v
    if p.kind in ("choice", "color") and p.choices:
        i = p.choices.index(v) if v in p.choices else -1
        return p.choices[(i + 1) % len(p.choices)]
    return v


def hotkeys(mode: str) -> dict[str, tuple[str, int]]:
    """Key character -> (param key, direction) for params active in *mode*."""
    out: dict[str, tuple[str, int]] = {}
    for p in PARAMS:
        if p.mode not in (None, mode):
            continue
        if len(p.keys) == 2:
            out[p.keys[0]] = (p.key, -1)
            out[p.keys[1]] = (p.key, +1)
        elif len(p.keys) == 1:
            out[p.keys[0]] = (p.key, +1)
    return out


def interpolate(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    """Blend two value dicts: numbers and colours glide, everything else switches at t = 0.5."""
    out = dict(a)
    for p in PARAMS:
        if p.key not in a or p.key not in b:
            continue
        va, vb = a[p.key], b[p.key]
        if p.kind == "float":
            out[p.key] = va + (vb - va) * t
        elif p.kind == "int":
            out[p.key] = int(round(va + (vb - va) * t))
        elif p.kind == "color":
            ca, cb = hex_to_bgr(va), hex_to_bgr(vb)
            out[p.key] = bgr_to_hex(tuple(x + (y - x) * t for x, y in zip(ca, cb)))
        else:
            out[p.key] = vb if t >= 0.5 else va
    return out
