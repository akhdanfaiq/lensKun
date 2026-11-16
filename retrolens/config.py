"""Central place for tunable constants. Colours are BGR (OpenCV order)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CAPTURE_DIR = ROOT / "captures"

HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
SEGMENT_MODEL = MODEL_DIR / "selfie_segmenter.tflite"

WINDOW_TITLE = "RETROLENS 2.0"

# --- palette -----------------------------------------------------------------
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (120, 255, 60)
ORANGE = (0, 150, 255)
RED = (60, 60, 255)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK = (24, 12, 20)

BRUSH_COLORS = [
    ("CYAN", CYAN),
    ("MAGENTA", MAGENTA),
    ("YELLOW", YELLOW),
    ("LIME", GREEN),
    ("ORANGE", ORANGE),
    ("WHITE", WHITE),
]

# --- tracking ----------------------------------------------------------------
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6
# One Euro filter: lower min_cutoff = smoother when still, higher beta = less lag when moving.
SMOOTH_MIN_CUTOFF = 1.4
SMOOTH_BETA = 0.012

# --- gestures (distances are relative to hand size, so they work near or far) -
PINCH_RATIO = 0.38          # thumb-pinky distance / hand size
INDEX_TOUCH_RATIO = 0.40    # index-index distance / average hand size
GESTURE_COOLDOWN = 0.45     # seconds between two "next" triggers
PEACE_HOLD = 0.8            # seconds a peace sign must be held to start the photo timer

# --- portal ------------------------------------------------------------------
PORTAL_GRACE = 0.25         # keep the portal alive this long when tracking drops out
PORTAL_OPEN_TIME = 0.28     # open animation duration
PORTAL_MIN_AREA = 1800      # px^2, below this the portal is considered closed
FILTER_TRANSITION = 0.35    # wipe duration when switching filters

# --- capture -----------------------------------------------------------------
PHOTO_COUNTDOWN = 3.0
RECORD_FPS = 30.0
