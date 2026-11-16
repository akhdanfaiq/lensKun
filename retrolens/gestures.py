"""Turn raw hand landmarks into poses and one-shot gesture events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import config
from . import tracking as tk
from .tracking import Hand

# Poses
POINT, PEACE, PALM, FIST, OTHER = "point", "peace", "palm", "fist", "other"
# Events
EVENT_NEXT, EVENT_PHOTO = "next", "photo"


def fingers_up(hand: Hand) -> Tuple[bool, bool, bool, bool, bool]:
    """(thumb, index, middle, ring, pinky). Rotation-invariant: compares distances to the wrist."""
    wrist = hand.pts[tk.WRIST]

    def extended(tip: int, pip: int) -> bool:
        return np.linalg.norm(hand.pts[tip] - wrist) > 1.12 * np.linalg.norm(hand.pts[pip] - wrist)

    thumb = hand.dist(tk.THUMB_TIP, tk.PINKY_MCP) > 1.08 * hand.dist(tk.THUMB_IP, tk.PINKY_MCP)
    return (
        thumb,
        extended(tk.INDEX_TIP, tk.INDEX_PIP),
        extended(tk.MIDDLE_TIP, tk.MIDDLE_PIP),
        extended(tk.RING_TIP, tk.RING_PIP),
        extended(tk.PINKY_TIP, tk.PINKY_PIP),
    )


def classify(hand: Hand) -> str:
    thumb, index, middle, ring, pinky = fingers_up(hand)
    if index and middle and ring and pinky:
        return PALM
    if index and middle and not ring and not pinky:
        return PEACE
    if index and not middle and not ring and not pinky:
        return POINT
    if not (index or middle or ring or pinky):
        return FIST
    return OTHER


@dataclass
class GestureFrame:
    events: Set[str] = field(default_factory=set)
    poses: Dict[str, str] = field(default_factory=dict)
    peace_progress: float = 0.0                 # 0..1 while a peace sign is being held
    peace_anchor: Optional[Tuple[int, int]] = None


class GestureEngine:
    def __init__(self):
        self._next_active = False
        self._next_time = -1e9
        self._peace_since: Optional[float] = None
        self._peace_fired = False

    def update(self, hands: List[Hand], t: float) -> GestureFrame:
        out = GestureFrame()
        for hand in hands:
            out.poses[hand.key] = classify(hand)

        # --- "next" : thumb + pinky pinch, or both index fingertips touching ----
        next_now = False
        for hand in hands:
            _, index, middle, _, _ = fingers_up(hand)
            # Requiring index + middle up stops a closed fist from counting as a pinch.
            if index and middle and hand.dist(tk.THUMB_TIP, tk.PINKY_TIP) < config.PINCH_RATIO * hand.size:
                next_now = True
        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            gap = float(np.linalg.norm(a.pts[tk.INDEX_TIP] - b.pts[tk.INDEX_TIP]))
            if gap < config.INDEX_TOUCH_RATIO * 0.5 * (a.size + b.size):
                next_now = True
        if next_now and not self._next_active and t - self._next_time > config.GESTURE_COOLDOWN:
            out.events.add(EVENT_NEXT)
            self._next_time = t
        self._next_active = next_now

        # --- "photo" : hold a peace sign ---------------------------------------
        peace_hand = next((h for h in hands if out.poses[h.key] == PEACE), None)
        if peace_hand is None:
            self._peace_since, self._peace_fired = None, False
        else:
            if self._peace_since is None:
                self._peace_since = t
            progress = min(1.0, (t - self._peace_since) / config.PEACE_HOLD)
            if not self._peace_fired:
                out.peace_progress = progress
                tip = peace_hand.pts[[tk.INDEX_TIP, tk.MIDDLE_TIP]].mean(axis=0)
                out.peace_anchor = (int(tip[0]), int(tip[1]))
                if progress >= 1.0:
                    out.events.add(EVENT_PHOTO)
                    self._peace_fired = True
                    out.peace_progress = 0.0
        return out
