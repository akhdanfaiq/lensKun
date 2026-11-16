"""Photos (with countdown + flash) and video recording into the captures/ folder."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

FLASH_TIME = 0.3


class CaptureManager:
    def __init__(self, out_dir: Path = config.CAPTURE_DIR):
        self.out_dir = Path(out_dir)
        self._countdown_end: Optional[float] = None
        self._flash_at = -1e9
        self._writer = None
        self._rec_start = 0.0
        self._rec_frames = 0
        self.last_path: Optional[Path] = None

    def _path(self, ext: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"retrolens_{stamp}.{ext}"
        n = 2
        while path.exists():
            path = self.out_dir / f"retrolens_{stamp}_{n}.{ext}"
            n += 1
        return path

    # --- photo -------------------------------------------------------------------
    def start_countdown(self, t: float) -> None:
        if self._countdown_end is None:
            self._countdown_end = t + config.PHOTO_COUNTDOWN

    def countdown_left(self, t: float) -> Optional[float]:
        return None if self._countdown_end is None else max(0.0, self._countdown_end - t)

    def flash_alpha(self, t: float) -> float:
        return max(0.0, 1.0 - (t - self._flash_at) / FLASH_TIME)

    def snap(self, frame: np.ndarray, t: float) -> Optional[Path]:
        path = self._path("png")
        self._flash_at = t
        if not cv2.imwrite(str(path), frame):
            return None
        self.last_path = path
        return path

    # --- video -------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._writer is not None

    def record_elapsed(self, t: float) -> float:
        return t - self._rec_start if self.recording else 0.0

    def toggle_recording(self, frame: np.ndarray, t: float) -> Optional[Path]:
        """Start or stop. Returns the file path on both start and stop, None if start failed."""
        if self.recording:
            self._writer.release()
            self._writer = None
            return self.last_path
        path = self._path("mp4")
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), config.RECORD_FPS, (w, h))
        if not writer.isOpened():
            return None
        self._writer, self._rec_start, self._rec_frames = writer, t, 0
        self.last_path = path
        return path

    # --- per frame -----------------------------------------------------------------
    def update(self, frame: np.ndarray, t: float) -> Optional[Path]:
        """Feed the clean (HUD-free) frame. Returns a path when a countdown photo was just taken."""
        if self._writer is not None:
            # Repeat frames as needed so playback speed matches real time at any app FPS.
            due = int((t - self._rec_start) * config.RECORD_FPS) + 1
            for _ in range(min(max(due - self._rec_frames, 0), 8)):
                self._writer.write(frame)
            self._rec_frames = max(self._rec_frames, due)
        if self._countdown_end is not None and t >= self._countdown_end:
            self._countdown_end = None
            return self.snap(frame, t)
        return None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
