"""Monotonic window scheduler.

Yields successive deadlines for a fixed-period loop. Uses ``time.monotonic``
so it survives wall-clock jumps.
"""
from __future__ import annotations

import time
from typing import Iterator


def window_deadlines(period_s: float, start: float | None = None) -> Iterator[float]:
    """Yield ``start, start+period, start+2*period, ...`` monotonic timestamps.

    The caller sleeps until each yielded deadline, then does its work and
    advances. If work overruns, the next deadline is in the past and the
    loop should emit a ``window_stalled`` health event before advancing.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    t = start if start is not None else time.monotonic()
    while True:
        yield t
        t += period_s


def sleep_until(deadline: float) -> float:
    """Sleep until ``deadline`` (monotonic). Returns how much the loop overran by
    (0 if on time, positive if behind).
    """
    now = time.monotonic()
    if now < deadline:
        time.sleep(deadline - now)
        return 0.0
    return now - deadline
