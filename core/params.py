"""Every tunable option, declared once.

The Align Perform look panel, its hotkeys, scene files and look crossfades are
all generated from PARAMS, so adding an option is one new Param(...) line plus
the code that reads it (core.pipeline.apply_params).

Values live in plain dicts {key: value} so they serialise to JSON as-is.
Colours are '#rrggbb' strings; convert with core.filters.hex_to_bgr.

Most options belong to the look (saved in looks, crossfaded).  A few are show
settings (Param.look = False): tuned once for the site and its hardware, never
saved in looks and never changed by a crossfade or the idle look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.filters import BG_PRESETS, FG_PRESETS, bgr_to_hex, hex_to_bgr
from core.posterize import PALETTE_STYLES

KINDS = ("int", "float", "bool", "choice", "color", "path", "look")
MODES = ("colour", "depth")


@dataclass(frozen=True)
class Param:
    """One option.

    Attributes:
        key:      Name used in dicts, JSON and code.
        label:    Text shown in the UI.
        group:    UI section.
        kind:     One of KINDS ("look": name of one of the scene's looks, "" = none).
        default:  Default value.
        min, max: Range for numbers.
        step:     Hotkey / spin-box increment for numbers.
        choices:  Options for "choice"; presets cycled by the hotkey for "color".
        keys:     Hotkeys: (decrease, increase) for numbers, (toggle,) / (cycle,) otherwise.
        mode:     Only relevant in this mode ("colour" / "depth"); None = always.
        help:     Tooltip.
        look:     Part of looks; False = show setting (see the module docstring).
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
    look: bool = True

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
    Param("seamless", "Seamless colours", "Tree", "bool", False, keys=("j",),
          help="Different colours meet only along ornament edges without ink: a triangle whose inked "
               "edge touches another colour splits (down to Max depth), and the smallest ones share "
               "a colour. Best with polarity alternate and few colours"),
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
    # (input filters only shape colour mode; depth mode thresholds the raw frames)
    Param("blur", "Blur", "Input", "int", 0, 0, 20, 1, keys=("[", "]"), mode="colour"),
    Param("sharpen", "Sharpen", "Input", "float", 0.0, 0.0, 5.0, 0.2, keys=(";", "'"), mode="colour"),
    Param("brightness", "Brightness", "Input", "int", 0, -100, 100, 10, keys=("b", "B"), mode="colour"),
    Param("contrast", "Contrast", "Input", "float", 1.0, 0.1, 3.0, 0.1, keys=("n", "N"), mode="colour"),
    Param("saturation", "Saturation", "Input", "float", 1.0, 0.0, 3.0, 0.1, keys=("t", "T"), mode="colour"),

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
          help="Pixels per stage cell used for colour and motion statistics (show setting: costs CPU)",
          look=False),

    # --- Presence ----------------------------------------------------------
    # Moving people are found against a learned picture of the empty stage
    Param("presence", "People only", "Presence", "bool", False, keys=("o",),
          help="Full ornament only where people move; still areas fade and coarsen"),
    Param("still_dim", "Still areas fade", "Presence", "float", 0.7, 0.0, 1.0, 0.05,
          help="How far still areas fade towards the background colour (1 = gone)"),
    Param("still_depth", "Still areas depth", "Presence", "int", 2, 0, 12, 1,
          help="Deepest split level where nobody moves"),
    Param("presence_trail", "Trail (s)", "Presence", "float", 1.0, 0.0, 10.0, 0.1,
          help="How long detail lingers where someone has just been"),
    Param("presence_palette", "Colours from people", "Presence", "bool", True, mode="colour",
          help="Fit the posterize colours to the people, so clothes keep their colours"),
    Param("presence_threshold", "Motion threshold", "Presence", "int", 24, 4, 100, 1,
          help="Colour difference (0-255) from the empty stage that counts as a person; "
               "lower is more sensitive (show setting)", look=False),
    Param("presence_absorb", "Absorb still after (s)", "Presence", "float", 60.0, 0.0, 600.0, 5.0,
          help="Someone or something that stays perfectly still this long becomes background; "
               "0 = never (show setting)", look=False),
    Param("presence_min_size", "Smallest person (%)", "Presence", "float", 1.0, 0.1, 20.0, 0.1,
          help="Moving areas smaller than this share of a cell are noise (show setting)", look=False),

    # --- Waves -------------------------------------------------------------
    Param("waves", "Waves", "Waves", "bool", False, keys=("w",),
          help="Rings of finer triangles spread from where people stand"),
    Param("wave_interval", "Pulse every (s)", "Waves", "float", 2.5, 0.0, 10.0, 0.1,
          help="Each person sends a new ring this often; 0 = only when they arrive"),
    Param("wave_origin", "Ring centre", "Waves", "choice", "feet", choices=("feet", "body"),
          help="feet: bottom of the person's outline (front cameras); body: its centre (top-down cameras)"),
    Param("wave_speed", "Speed (cells/s)", "Waves", "float", 0.6, 0.05, 5.0, 0.05),
    Param("wave_range", "Range (cells)", "Waves", "float", 1.5, 0.2, 10.0, 0.1,
          help="Distance a ring travels before it has faded"),
    Param("wave_width", "Ring width (cells)", "Waves", "float", 0.12, 0.02, 1.0, 0.01),
    Param("wave_depth", "Ring detail (+levels)", "Waves", "int", 3, 0, 8, 1,
          help="Triangles under a ring's crest split this many more times"),
    Param("wave_glow", "Glow", "Waves", "float", 0.6, 0.0, 1.0, 0.05,
          help="How far the ornament under a ring tints towards the glow colour"),
    Param("wave_color", "Glow colour", "Waves", "color", "#b4232d"),

    # --- Idle --------------------------------------------------------------
    Param("idle_look", "Idle look", "Idle", "look", "",
          help="Crossfade to this look while nobody is on stage (show setting)", look=False),
    Param("idle_after", "After nobody for (s)", "Idle", "float", 20.0, 1.0, 600.0, 1.0, look=False,
          help="Seconds without anyone on stage before the idle look starts (show setting)"),
    Param("idle_fade", "Idle fade (s)", "Idle", "float", 3.0, 0.0, 30.0, 0.5, look=False,
          help="Crossfade time into and out of the idle look (show setting)"),
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
        return str(value)          # path, look
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


def look_values(values: dict[str, Any]) -> dict[str, Any]:
    """The part of *values* a look stores (show settings left out)."""
    return {k: v for k, v in values.items() if k in BY_KEY and BY_KEY[k].look}


def look_target(current: dict[str, Any], look: dict[str, Any]) -> dict[str, Any]:
    """Values after switching from *current* to *look*: the look's values, the current show settings."""
    return normalise(dict(current, **look_values(look)))


def interpolate(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    """Blend two value dicts: numbers and colours glide, everything else switches at t = 0.5.

    Show settings (Param.look = False) keep a's value.
    """
    out = dict(a)
    for p in PARAMS:
        if p.key not in a or p.key not in b or not p.look:
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
