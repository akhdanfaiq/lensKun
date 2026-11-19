# Retrolens 2.0 — Hand-Tracking Portal, Air Drawing & Capture

An interactive webcam toy built with **MediaPipe** and **OpenCV**. Frame a "portal" between your
fingers to look through 18 live filters, draw neon strokes in the air, and save photos or videos —
all controlled by hand gestures.

Inspired by [GedeAnanda/python-handtrack](https://github.com/GedeAnanda/python-handtrack), rebuilt
as a modular project with smoother tracking, more effects and a cyberpunk HUD.

## Features

- **Portal** — a glowing, animated quad between the thumb and index fingertips of both hands,
  with sparks, an opening animation and a wipe transition when the filter changes.
- **18 filters** — Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon,
  Galaxy (background replacement), VHS, Matrix, Hologram, Cartoon, Mirror, Cyberpunk, Night Vision.
- **Smooth tracking** — landmarks pass through a One Euro filter, so the portal does not jitter.
- **Scale-aware gestures** — thresholds are relative to hand size, so they work near or far.
- **Air drawing mode** — paint neon strokes with your index finger, erase with an open palm.
- **Photo & video capture** — 3-second countdown with flash, MP4 recording, saved to `captures/`
  without the HUD or hand skeleton.
- **Themes** — a clean *Minimal* look by default, or a neon *Cyberpunk* look; press `T` to switch.

## Requirements

- Python 3.9 or newer (tested on 3.14, macOS)
- A webcam

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py        # only needed if the models/ folder is empty
```

## Run

```bash
python main.py
```

Options: `--camera 1` (other camera), `--width 1280 --height 720`, `--no-mirror`,
`--source clip.mp4` (use a video or image file instead of the webcam).

> **macOS:** the first run asks for camera permission for your terminal app. If the camera does not
> open, enable it in *System Settings → Privacy & Security → Camera* and run again.

## Controls

| Gesture | Action |
| --- | --- |
| Thumb + index of **both** hands in view | Open the portal between the four fingertips |
| Touch thumb and pinky (index + middle up) | Next filter (next colour in draw mode) |
| Touch both index fingertips together | Next filter (next colour in draw mode) |
| Hold a peace sign for ~1 second | Start the photo countdown |
| Point with the index finger (draw mode) | Draw |
| Open palm (draw mode) | Erase under the palm |

| Key | Action |
| --- | --- |
| `N` / `P` | Next / previous filter or colour |
| `D` or `Tab` | Switch between portal and draw mode |
| `T` | Switch theme (Minimal / Cyberpunk) |
| `Space` | Photo (3 s countdown) |
| `R` | Start / stop video recording |
| `C` | Clear the drawing |
| `H` | Show / hide help |
| `Q` or `Esc` | Quit |

## Project structure

```
main.py               entry point: camera loop, window, keyboard
download_models.py    fetches the MediaPipe models
retrolens/
  config.py           all tunable constants (colours, thresholds, timings)
  tracking.py         MediaPipe hand landmarker + segmenter, One Euro smoothing
  gestures.py         finger states, poses, one-shot gesture events
  filters.py          the 18 portal filters
  portal.py           portal geometry, glow, particles, transitions
  canvas.py           air drawing
  capture.py          photo countdown and video recording
  hud.py              on-screen interface
  theme.py            visual themes (Minimal, Cyberpunk)
  fx.py               shared drawing helpers (glow, panels, text)
  app.py              ties everything together
models/               hand_landmarker.task, selfie_segmenter.tflite
captures/             saved photos and videos (created on first capture)
```

### Themes

| Theme | Look |
| --- | --- |
| `minimal` (default) | White and grey text on soft dark panels, thin portal outline, faint hand skeleton, muted brush colours |
| `cyberpunk` | Neon cyan/magenta glow, sparks around the portal, corner brackets, neon brush strokes |

To change the theme the app starts with, set `THEME` in `retrolens/config.py`. To make your own,
add a `Theme(...)` to `THEMES` in `retrolens/theme.py` — colours, panel style, glow on/off and the
brush palette are all defined there.

### Adding a filter

Write a function in `retrolens/filters.py` that takes `(roi, ctx)` and returns a new BGR image of
the same size, then add it to `build_filters()`:

```python
def my_filter(roi, ctx):
    return cv2.flip(roi, 0)

Filter("UPSIDE DOWN", my_filter)
```

## Tips

- Good, even lighting makes hand detection far more reliable.
- Keep both hands fully inside the frame when opening the portal.
- If the Galaxy filter shows a double exposure instead of replacing the background,
  `models/selfie_segmenter.tflite` is missing — run `python download_models.py`.
