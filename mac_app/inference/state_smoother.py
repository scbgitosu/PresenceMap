"""3-of-5 hysteresis smoother for live occupancy predictions.

Tracks the last ``window_size`` raw predictions. Transitions the stable state
only when ``threshold`` of the last ``window_size`` predictions agree on a
different state. Bootstraps with the first prediction until enough windows
have arrived.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class SmoothedState:
    raw: str
    stable: str
    confidence: float
    motion_intensity: float
    last_changed_at: Optional[str] = None


class StateSmoother:
    def __init__(self, window_size: int = 5, threshold: int = 3) -> None:
        if threshold > window_size:
            raise ValueError("threshold > window_size")
        self.window_size = window_size
        self.threshold = threshold
        self._history: Deque[str] = deque(maxlen=window_size)
        self._stable: Optional[str] = None
        self._last_change_ts: Optional[str] = None

    @property
    def stable(self) -> str:
        return self._stable or "unknown"

    def update(
        self,
        raw_pred: str,
        confidence: float,
        motion_intensity: float,
        *,
        ts: Optional[str] = None,
    ) -> SmoothedState:
        self._history.append(raw_pred)
        if self._stable is None:
            self._stable = raw_pred
            self._last_change_ts = ts
        elif len(self._history) >= self.window_size:
            counts = Counter(self._history)
            most_common, n = counts.most_common(1)[0]
            if most_common != self._stable and n >= self.threshold:
                self._stable = most_common
                self._last_change_ts = ts
        return SmoothedState(
            raw=raw_pred,
            stable=self._stable,
            confidence=confidence,
            motion_intensity=motion_intensity,
            last_changed_at=self._last_change_ts,
        )
