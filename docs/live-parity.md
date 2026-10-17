# live.py → Align Perform / perform.py

Every control in `live.py` and where it lives now. "Hotkey" keys work on the Align Perform canvas (when it has focus) and on projector windows, exactly as in `live.py`.

## Keys

| live.py | What | Align Perform | perform.py |
|---|---|---|---|
| `d` / `D` | max depth | Look › Tree › Max depth, hotkey | hotkey |
| `m` / `M` | min depth | Look › Tree › Min depth, hotkey | hotkey |
| `h` / `H` | split threshold | Look › Tree › Split threshold, hotkey | hotkey |
| `c` / `C` | colours | Look › Colour › Colours, hotkey | hotkey |
| `p` / `P` | statistics resolution | Look › Render › Statistics resolution, hotkey | hotkey |
| `o` / `O` | output resolution | Inspector › Projector › Resolution (each projector renders at its own resolution) | from the scene |
| `[` / `]` | blur | Look › Input › Blur, hotkey | hotkey |
| `;` / `'` | sharpen | Look › Input › Sharpen, hotkey | hotkey |
| `n` / `N` | contrast | Look › Input › Contrast, hotkey | hotkey |
| `t` / `T` | saturation | Look › Input › Saturation, hotkey | hotkey |
| `b` / `B` | brightness (colour mode) | Look › Input › Brightness, hotkey | hotkey |
| `b` | background colour (depth mode) | Look › Depth › Background, hotkey cycles presets | hotkey |
| `f` | foreground colour (depth mode) | Look › Depth › Foreground, hotkey cycles presets | hotkey |
| `a` / `A`, `z` / `Z` | depth band | Look › Depth › Band min / max, hotkeys | hotkeys |
| `k` | palette style | Look › Colour › Palette style, hotkey | hotkey |
| `v` | conforming mesh | Look › Tree › Conforming mesh, hotkey | hotkey |
| `i` | animation | Look › Tree › Animate, hotkey | hotkey |
| `x` | ornament polarity | Look › Render › Polarity, hotkey | hotkey |
| `g` | cord | Look › Render › Cord, hotkey | hotkey |
| `e` | wireframe | Look › Render › Wireframe, hotkey | hotkey |
| `r` | mirror | Inspector › Camera › Mirror horizontally (per camera) | from the scene |
| `s` | save frame | File › Export snapshot… (Ctrl+E): every projector + the stage | `S` |
| `u` | HUD | status bar (always); panels via View menu | `H` status overlay |
| `w` / `F11` / `Esc` | fullscreen | Output › Projector outputs (F5) opens every projector full screen on its display | always full screen (`--windowed` for testing) |
| `q` | quit | File › Quit (Ctrl+Q) | `Ctrl+Shift+Q` |

## Command-line options

| live.py | Align Perform |
|---|---|
| `source` (webcam / video / GIF / image) | Add › Camera… (webcam list or file); any number of cameras |
| `--depth` | Look › Mode › depth |
| `--out` | each projector's resolution |
| `--proc` | Look › Render › Statistics resolution (per stage cell) |
| `--levels` | Inspector › Stage › Levels |
| `--display` | n/a (canvas zooms) |
| `--crop` | Look › Render › Crop tile margin |
| `--fullscreen` | Output › Projector outputs |
| `--mirror` / `--no-mirror` | Inspector › Camera › Mirror horizontally |

## New in Align Perform

Ornament tile file, background colour, merge / colour hysteresis, split time and smoothing time constants, palette smoothing and label margin (all in the Look panel); per-camera brightness / contrast / saturation and edge feather; projector gain, edge-blend width and gamma; looks with crossfades.
