"""MediaPipe wrappers: smoothed hand landmarks and person segmentation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from . import config

# Landmark indices (MediaPipe hand model).
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class Hand:
    key: str            # stable id across frames ("Left", "Right", "Left#2", ...)
    pts: np.ndarray     # (21, 2) float32 pixel coordinates, smoothed

    @property
    def size(self) -> float:
        """Palm length in pixels; used to make gesture thresholds scale-invariant."""
        return float(np.linalg.norm(self.pts[MIDDLE_MCP] - self.pts[WRIST])) + 1e-6

    @property
    def palm(self) -> np.ndarray:
        return self.pts[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)

    def dist(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.pts[a] - self.pts[b]))


class OneEuroFilter:
    """Adaptive low-pass filter: kills jitter when still, stays responsive when moving."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None
        self.t = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(dt, cutoff)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x


class HandTracker:
    def __init__(self):
        if not config.HAND_MODEL.exists():
            raise FileNotFoundError(
                f"Missing model: {config.HAND_MODEL}\n"
                "Run `python download_models.py` to fetch it."
            )
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(config.HAND_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=config.MAX_HANDS,
                min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.DETECTION_CONFIDENCE,
                min_tracking_confidence=config.TRACKING_CONFIDENCE,
            )
        )
        self._segmenter = None
        if config.SEGMENT_MODEL.exists():
            try:
                self._segmenter = vision.ImageSegmenter.create_from_options(
                    vision.ImageSegmenterOptions(
                        base_options=base(model_asset_path=str(config.SEGMENT_MODEL)),
                        running_mode=vision.RunningMode.VIDEO,
                        output_category_mask=False,
                        output_confidence_masks=True,
                    )
                )
            except Exception as exc:  # segmentation is optional
                print(f"[retrolens] segmentation disabled: {exc}")

        self._filters: Dict[str, OneEuroFilter] = {}
        self._last_seen: Dict[str, float] = {}
        self._last_ts = -1
        self._image = None
        self._shape = (0, 0)

    @property
    def has_segmenter(self) -> bool:
        return self._segmenter is not None

    def _timestamp(self, t: float) -> int:
        # MediaPipe VIDEO mode requires strictly increasing millisecond timestamps.
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def detect(self, frame_bgr: np.ndarray, t: float) -> List[Hand]:
        h, w = frame_bgr.shape[:2]
        self._shape = (h, w)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts = self._timestamp(t)
        result = self._landmarker.detect_for_video(self._image, self._ts)

        raw = []
        for i, lms in enumerate(result.hand_landmarks or []):
            label = "Hand"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name or "Hand"
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            raw.append((label, pts))
        raw.sort(key=lambda item: float(item[1][WRIST][0]))

        hands: List[Hand] = []
        used = set()
        for label, pts in raw:
            key, n = label, 2
            while key in used:
                key, n = f"{label}#{n}", n + 1
            used.add(key)
            # Restart smoothing if this hand has been gone for a moment.
            if t - self._last_seen.get(key, -1e9) > 0.3:
                self._filters.pop(key, None)
            smoother = self._filters.setdefault(
                key, OneEuroFilter(config.SMOOTH_MIN_CUTOFF, config.SMOOTH_BETA)
            )
            self._last_seen[key] = t
            hands.append(Hand(key=key, pts=smoother(pts, t).astype(np.float32)))
        return hands

    def segment(self) -> Optional[np.ndarray]:
        """Person probability mask (float32, 0..1, frame-sized) for the last detected frame."""
        if self._segmenter is None or self._image is None:
            return None
        result = self._segmenter.segment_for_video(self._image, self._ts)
        if not result.confidence_masks:
            return None
        mask = result.confidence_masks[0].numpy_view()
        mask = np.ascontiguousarray(mask.reshape(mask.shape[0], mask.shape[1]))
        h, w = self._shape
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(mask, (7, 7), 0)

    def close(self) -> None:
        self._landmarker.close()
        if self._segmenter is not None:
            self._segmenter.close()
