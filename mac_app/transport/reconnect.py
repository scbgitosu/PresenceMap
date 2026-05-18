"""Signal-loss watchdog for the Mac side.

Tracks the wall-clock age of the last received WindowMsg; if it exceeds
``stall_threshold_s`` the watchdog flips into ``signal_lost`` state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Watchdog:
    stall_threshold_s: float = 6.0
    _last_seen_monotonic: float = 0.0
    _signal_lost: bool = True  # start as 'lost' until first message arrives

    def mark_received(self) -> None:
        self._last_seen_monotonic = time.monotonic()
        self._signal_lost = False

    def age_s(self) -> float:
        if self._last_seen_monotonic == 0.0:
            return float("inf")
        return time.monotonic() - self._last_seen_monotonic

    def is_signal_lost(self) -> bool:
        if self.age_s() > self.stall_threshold_s:
            self._signal_lost = True
        return self._signal_lost
