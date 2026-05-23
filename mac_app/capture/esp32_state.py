"""Shared runtime state for ESP32 ingest (JSON on disk for Streamlit)."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

from shared.project import runtime_dir


@dataclass
class NodeSnapshot:
    node_id: int
    source_ip: str
    status: str  # ok | stale | offline
    last_seen_s_ago: float
    frames_total: int
    pps_1s: float
    sequence: int
    seq_gaps: int
    seq_gap_ratio: float
    rssi_dbm: int
    n_subcarriers: int
    n_antennas: int
    channel_freq_mhz: int
    frames_last_window: int = 0
    loss_ratio_last_window: float = 0.0


@dataclass
class AggregateSnapshot:
    total_pps: float
    nodes_ok: int
    nodes_stale: int
    nodes_offline: int
    csi_frames_parsed: int
    sibling_packets_skipped: int
    parse_errors: int
    windows_emitted: int


@dataclass
class PpsHistoryPoint:
    t: float
    pps: float


@dataclass
class FusedWindowSnapshot:
    window_id: str
    ts_end: float
    features: List[float]
    frames: int
    node_count: int


@dataclass
class Esp32RuntimeState:
    updated_at: str
    ingest_running: bool
    udp_port: int
    window_seconds: float
    nodes: List[NodeSnapshot] = field(default_factory=list)
    aggregate: Optional[AggregateSnapshot] = None
    pps_history: Dict[str, List[dict]] = field(default_factory=dict)
    recent_log: List[str] = field(default_factory=list)
    fused_window: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Esp32StateStore:
    """Thread-safe state + atomic JSON writes for cross-process dashboard reads."""

    def __init__(
        self,
        repo_root: Path,
        *,
        history_seconds: float = 60.0,
        max_log_lines: int = 50,
    ) -> None:
        self._path = runtime_dir(repo_root) / "esp32_state.json"
        self._lock = threading.Lock()
        self._history_seconds = history_seconds
        self._max_log = max_log_lines
        self._pps_history: Dict[str, Deque[PpsHistoryPoint]] = {}
        self._log: Deque[str] = deque(maxlen=max_log_lines)
        self._ingest_running = False
        self._udp_port = 5005
        self._window_seconds = 2.0
        self._nodes: Dict[str, NodeSnapshot] = {}
        self._fused_window: Optional[FusedWindowSnapshot] = None
        self._aggregate = AggregateSnapshot(
            total_pps=0.0,
            nodes_ok=0,
            nodes_stale=0,
            nodes_offline=0,
            csi_frames_parsed=0,
            sibling_packets_skipped=0,
            parse_errors=0,
            windows_emitted=0,
        )

    @property
    def path(self) -> Path:
        return self._path

    def set_meta(self, *, running: bool, udp_port: int, window_seconds: float) -> None:
        with self._lock:
            self._ingest_running = running
            self._udp_port = udp_port
            self._window_seconds = window_seconds

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(f"{time.strftime('%H:%M:%S')} {line}")

    def update_fused_window(self, fused: Optional[FusedWindowSnapshot]) -> None:
        with self._lock:
            self._fused_window = fused

    def update_nodes(self, nodes: List[NodeSnapshot], agg: AggregateSnapshot) -> None:
        with self._lock:
            self._nodes = {f"{n.source_ip}:{n.node_id}": n for n in nodes}
            self._aggregate = agg
            now = time.time()
            for n in nodes:
                key = f"{n.source_ip}:{n.node_id}"
                hist = self._pps_history.setdefault(key, deque())
                hist.append(PpsHistoryPoint(t=now, pps=n.pps_1s))
                cutoff = now - self._history_seconds
                while hist and hist[0].t < cutoff:
                    hist.popleft()

    def flush(self) -> None:
        with self._lock:
            snap = Esp32RuntimeState(
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ingest_running=self._ingest_running,
                udp_port=self._udp_port,
                window_seconds=self._window_seconds,
                nodes=sorted(self._nodes.values(), key=lambda n: (n.node_id, n.source_ip)),
                aggregate=self._aggregate,
                pps_history={
                    k: [{"t": p.t, "pps": p.pps} for p in v]
                    for k, v in self._pps_history.items()
                },
                recent_log=list(self._log),
                fused_window=asdict(self._fused_window) if self._fused_window else None,
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snap.to_dict(), indent=2))
        tmp.replace(self._path)

    @staticmethod
    def load(repo_root: Path) -> Optional[Esp32RuntimeState]:
        path = runtime_dir(repo_root) / "esp32_state.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Esp32RuntimeState(**data)
        except Exception:
            return None
