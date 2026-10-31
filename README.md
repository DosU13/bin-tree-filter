# bin-tree-filter

Real-time Kyrgyz ornament filter. Cameras (or video files) are turned into a mosaic of ornament tiles laid out on an adaptive triangle tree, and projected, seamlessly across several projectors.

Three programs:

| Program | For |
|---|---|
| `live.py` | Quick single-camera window (the original tool) |
| `python -m align_perform` | **Align Perform**: set up a scene with any number of cameras and projectors, tune the look, save it |
| `perform.py` | Play a saved scene on stage / at an installation: full screen, unattended |

## Setup (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

A GPU with OpenGL 3.3 is needed for projector output (any NVIDIA / AMD card from the last decade). `requirements-dev.txt` adds pytest.

**Depth camera (optional): Orbbec Astra / Astra S.** It isn't a webcam; it is read through OpenNI2 with Orbbec's driver.

1. Install the Orbbec driver (`SensorDriver_V4.3.0.22.exe`), re-plug the camera and reboot. Windows lists it as *ORBBEC … Depth Sensor*.
2. Unpack the Astra SDK (`AstraSDK-v2.1.3-…-win64`) **next to the project folder**: its `bin` folder (with `OpenNI2.dll` and `orbbec.dll`) is found automatically. Elsewhere: copy that `bin` folder to `vendor\openni2` in the project, or set `OPENNI2_REDIST64` to it.
3. Plug the Astra **straight into the PC**, not into a USB hub (on a shared hub it fails with *USB transfer timeout*), and keep half a metre clear in front of it: anything a few cm away switches its laser off and all depth reads as nothing. It measures from about 0.4 m to 5 m.

OrbbecViewer (Orbbec SDK v1) does not see the older Astra S; that is expected.

## Workflow

### 1. Align (Align Perform)

```
.venv\Scripts\python -m align_perform scenes\example.json
```

1. **Stage.** Select *Stage* in the scene list. The ornament lives only on the stage: a grid of square *cells*; the cell size is the ornament's scale. *Fit stage to projectors* sizes it for you once projectors are placed.
2. **Cameras.** *Add › Camera…* (Ctrl+Shift+C): pick a webcam (press *Test source* to see its real resolution), an Astra on the *Depth camera* tab, or, for testing, a video / image / GIF file. An Astra's depth stream is a grey picture, near white and far dark, from *Near* to *Far* (adjust them live in the inspector's *Depth range*; *Clean up*, on by default, fills the shadow beside people and steadies the depth's noise). In depth mode (Look tab, *Mode*) *Band min / max* (0–255) pick the slice of that range that gets the ornament, and *Layers* cuts it into slices by distance, each its own colour: *Layer colours* *fade* runs from *Foreground* into the background with distance, a palette from its boldest colour nearest. For the whole room try Band 5–255 with 5 layers. Each camera is a rectangle on the canvas showing the part of the stage it sees. Switch the view to *Camera images* (Ctrl+4) and line the cameras up so overlapping parts match; *Stitched input* (Ctrl+2) shows the result. Per-camera brightness / contrast / saturation in the inspector evens out cameras that expose differently; *Edge feather* sets how wide overlapping cameras cross-fade.
3. **Projectors.** *Add › Projector…* (Ctrl+Shift+P): pick a display, or *Window (testing)* on a one-monitor PC. Each projector is a rectangle showing the part of the stage it lights. Where projectors overlap, they are blended automatically (*Edge blend*, *Gamma*, *Brightness gain* in the inspector).
4. **Physical alignment.** *Output › Projector outputs* (F5) opens every projector full screen. *Test pattern* (F6) projects a grid, names, the stage outline and other projectors' edges over a blended grey field: move and size the rectangles until the projected grid lines meet across projectors and the grey looks even in the overlaps. With the test pattern on, arrow keys on a projector move it (Shift: ×10). For angled surfaces tick *4-corner warp* and drag corners on the canvas, or press Tab on the projector to pick a corner and nudge it with the arrows. *Identify* (F7) shows each projector's number.
5. **Look.** The *Look* tab has every option of `live.py` (and more). With the canvas focused, `live.py`'s keys work too (d/D, h/H, c/C, g, x, …; F1 lists them). Save favourite settings in the *Looks* tab and crossfade between them. *Seamless colours* (`j`) keeps different colours from meeting along an inked ornament edge: where they would, the triangles split until the colour change falls on edges without ink (the smallest triangles share a colour instead); best with polarity *alternate* and few colours, as it adds detail along colour boundaries.
6. **People** (optional, see below). Turn on *People only* (`o`) and / or *Waves* (`w`), pick an *Idle look*, and with nobody in front of the cameras press *Run › Learn empty stage* (Ctrl+L).
7. **Save** (Ctrl+S). Media paths are stored relative to the scene file, so keep clips next to it.

Mouse: wheel zooms, middle-drag or Space+drag pans, drag to move, handles to resize (aspect locked to the camera / projector; Shift = free, Alt = from centre, Ctrl = no snapping), Alt+click selects the item underneath. Everything is undoable (Ctrl+Z / Ctrl+Y).

### People: presence, waves, idle look

So that people see *themselves* in the mosaic, the filter can find who is moving in front of the cameras. It compares each frame with a learned picture of the empty stage (about half a millisecond per cell; no AI model, so a cheap laptop copes).

- **People only** (`o`): full ornament only where people are. Still areas fade towards the background (*Still areas fade*) and keep big triangles (*Still areas depth*); the triangles along a person's outline split finely, so silhouettes stay crisp. *Trail* keeps detail for a moment where someone has just been. *Colours from people* fits the posterize colours to the people, so clothes keep their colours.
- **Waves** (`w`): rings spread from where each person stands, once when they arrive and then every *Pulse every* seconds. Triangles under a ring split *Ring detail* more levels and tint towards *Glow colour*. Use *Ring centre: feet* for cameras facing people, *body* for cameras looking down. Waves are drawn on the GPU, so they move smoothly at the projectors' frame rate.
- **Idle look**: crossfade to one of the scene's looks after nobody has been on stage for *After nobody for* seconds, and back as soon as someone comes. `perform.py` always runs it; in Align Perform turn on *Run › Run idle look* to try it.
- **Learn empty stage** (Ctrl+L in Align Perform, `L` in `perform.py`): take the next half second of frames as the empty stage. Do it with nobody in view, after the lights are set. The background also adapts on its own: slow lighting changes are followed, anything that stays perfectly still for *Absorb still after* seconds becomes background (a moved chair, someone who was there when it was learned), and if most of the stage suddenly differs for 2 s (lights switched, camera bumped) it is relearned.
- *View › People* (Ctrl+5) shows what is detected: red = presence and trails, circles = people, crosses = their feet. Tune *Motion threshold* (lower = more sensitive) and *Smallest person* there.

- **Motion glow** (`l`): movement lights the ornament up, and a small fluid simulation carries the light on, so a waving hand sends it further (*Flow*: how hard movement pushes it, 0 = it stays put; *Fade*: how long it lasts). *Light from*: *motion* lights whatever moves; *person* lights whoever is in front of the environment, even standing still: the environment is each pixel's most common value over the last *Environment memory* seconds (30 by default), so it learns itself while people play and needs no *Learn empty stage*; *both* lights the person dimmer and their movement full. It works in both modes; with a depth camera only what is nearer than the environment counts, and the changing projection shouldn't trigger it the way it can with a webcam. *Light threshold* sets how much change counts (depth: 8 ≈ 12 cm with a 4 m range). With *People only* on, lit areas also get full detail. The light takes the main colour of the palette in use (*Layer colours* / *Palette style*): the red of navy-red-gold, tush-kiyiz and shyrdak. That colour then lights only the glow: the layers or posterize colours use the rest. Without a palette (*fade*, *natural*) it is *Light colour*, white by default. It glows best on a dark background.

Names in italics in the *Look* tab (*Motion threshold*, *Absorb still after*, *Smallest person*, the idle settings, *Statistics resolution*, *Light threshold*, *Environment memory*, *Mirror fold*) are **show settings**: tuned once for the site, they are not saved in looks and never change in a crossfade.

Tips: lock the webcams' exposure and white balance if the driver allows it; keep the projection out of the cameras' view (the projected picture changing looks like movement); `scenes\people.json` is a ready example with a test clip of two walkers.

### Two projectors, one picture (mirror fold)

When the PC can only feed one picture to two projectors (a splitter, or Windows' *Duplicate these displays*), the second projector can mirror it, and *Mirror fold* makes that work:

1. Put the projectors side by side. Feed both the same picture and set the second one to mirror left-right in its own menu (often *Rear* projection). For projectors one above the other, use top-bottom (*Ceiling*) instead.
2. Make the stage as wide as both projected areas together, with the centre line where they meet (don't use *Fit stage to projectors*: it would fit the one projector). Place the camera(s) over the whole stage as usual.
3. Add **one** projector, on the first projector's half of the stage.
4. Look tab › Input › *Mirror fold*: *left-right* (or *top-bottom*).

The stage is laid over its mirror image: averaged in colour mode, the nearer of the two in depth mode. Whoever stands in front of either projector sees themselves in place there, with a mirrored ghost of whoever is at the other one.

### 2. Perform

```
.venv\Scripts\python perform.py scenes\show.json
.venv\Scripts\python perform.py scenes\show.json --look calm --status
.venv\Scripts\python perform.py scenes\example.json --windowed      (try it on one monitor)
```

| Key (on any projector) | |
|---|---|
| `live.py` keys | change the running look (not saved) |
| `1`–`9` / `0` | crossfade to the scene's looks / back to its own settings (`--fade` seconds) |
| `B` | blackout (fades) |
| `L` | learn the empty stage (step out of view first) |
| `F` | freeze |
| `R` | reload the scene file |
| `H` | status overlay |
| `T` / `I` | test pattern / identify |
| `S` | save every projector's image (`logs/snapshots/`) |
| `Ctrl+Shift+Q` | quit (Esc does nothing during a show) |

Unattended operation: cameras that disconnect are reopened (their last picture holds meanwhile), a stalled pipeline is restarted, the display is kept awake, and everything is logged to `logs/perform-YYYYMMDD.log` (a health line every minute, idle look changes). `--wait-for-screens` keeps a projector dark until its display appears instead of opening a window. If the scene has an idle look, it takes over while nobody is on stage.

**Start with Windows:** use `tools\perform_loop.bat scenes\show.json`, which restarts `perform.py` if it ever exits with an error. Put it in Task Scheduler: *Create Task › Triggers: At log on › Actions: Start a program* = the full path of `tools\perform_loop.bat`, *Add arguments* = the scene path, *Start in* = the project folder. Turn off Windows Update restarts and screen savers on show machines.

### live.py

```
.venv\Scripts\python live.py            (webcam 0)
.venv\Scripts\python live.py me.jpg
.venv\Scripts\python live.py depth.gif --depth
```

Keys are listed in `live.py`'s docstring. `docs/live-parity.md` maps each one to Align Perform and `perform.py`.

## Tools

| Command | |
|---|---|
| `python -m tools.render_scene scenes\example.json` | run a scene without UI; writes every projector's image and a simulated projection to `out/render/` |
| `python -m tools.bench --gpu` | per-stage timings, CPU and GPU |
| `python -m tools.make_test_clips` | regenerate the example's test clips (`--people`: the walkers clip and `scenes\people.json`) |
| `python -m pytest` | tests (`-m "not gui"` skips the ones that open windows) |

## How it works

`docs/perform-plan.md` is the design; `CLAUDE.md` is the developer guide.
