"""Per-window heartbeat + health-event helper.

Wraps the publisher with a small stateful counter so the ``window_loop`` can
emit ``window_stalled`` / ``target_ssid_low_visibility`` without each loop
having to maintain its own counters.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from hp_agent.transport.messages import make_health_msg
from hp_agent.transport.zmq_publisher import Publisher
from shared.utils import now_iso


@dataclass
class HealthState:
    target_seen_history: Deque[bool] = field(default_factory=lambda: deque(maxlen=60))
    last_low_visibility_emitted: bool = False


class HeartbeatEmitter:
    def __init__(self, publisher: Publisher, *, agent_id: str, session_id: str) -> None:
        self.publisher = publisher
        self.agent_id = agent_id
        self.session_id = session_id
        self.state = HealthState()

    def emit(self, code: str, detail: str = "", **metrics) -> None:
        msg = make_health_msg(
            agent_id=self.agent_id,
            session_id=self.session_id,
            ts=now_iso(),
            code=code,
            detail=detail,
            metrics=metrics,
        )
        self.publisher.send_health(msg)

    def record_window(
        self,
        *,
        target_seen: bool,
        loss_ratio: float,
        overrun_s: float,
        scan_error: Optional[str] = None,
    ) -> None:
        """Called once per window. Emits heartbeats + alarms when warranted."""
        self.state.target_seen_history.append(target_seen)
        self.emit(
            "ok",
            detail="heartbeat",
            target_seen=target_seen,
            loss_ratio=loss_ratio,
            overrun_s=overrun_s,
        )
        if scan_error:
            self.emit("scan_failed", detail=scan_error)
        if overrun_s > 0:
            self.emit("window_stalled", detail=f"overran deadline by {overrun_s:.2f}s")
        # Target-visibility alarm: at least half the window of 60 must be False.
        h = self.state.target_seen_history
        if len(h) >= 30:
            hit_rate = sum(1 for x in h if x) / len(h)
            low = hit_rate < 0.5
            if low and not self.state.last_low_visibility_emitted:
                self.emit(
                    "target_ssid_low_visibility",
                    detail=f"target visible in {hit_rate:.0%} of last {len(h)} windows",
                    hit_rate=hit_rate,
                )
                self.state.last_low_visibility_emitted = True
            elif not low and self.state.last_low_visibility_emitted:
                self.emit(
                    "ok",
                    detail=f"target visibility recovered to {hit_rate:.0%}",
                    hit_rate=hit_rate,
                )
                self.state.last_low_visibility_emitted = False
