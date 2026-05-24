"""UDP ingest service for ESP32 CSI (ADR-018) with per-node diagnostics."""
from __future__ import annotations

import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from mac_app.capture.csi_features import DEFAULT_NUM_SUBCARRIERS
from mac_app.capture.esp32_features import aggregate_esp32_window_features
from mac_app.capture.esp32_parser import Esp32CsiFrame, ParseError, ParseErrorKind, parse_stream
from mac_app.capture.esp32_state import (
    AggregateSnapshot,
    Esp32StateStore,
    FusedWindowSnapshot,
    NodeSnapshot,
)

OK_SECS = 2.0
STALE_SECS = 10.0


@dataclass
class _NodeTrack:
    node_id: int
    source_ip: str
    last_seen: float = 0.0
    frames_total: int = 0
    last_sequence: Optional[int] = None
    seq_gaps: int = 0
    seq_total: int = 0
    rssi_dbm: int = 0
    n_subcarriers: int = 0
    n_antennas: int = 0
    channel_freq_mhz: int = 0
    frame_times_1s: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    window_frames: List[Esp32CsiFrame] = field(default_factory=list)
    frames_last_window: int = 0
    loss_ratio_last_window: float = 0.0

    def key(self) -> str:
        return f"{self.source_ip}:{self.node_id}"

    def record_frame(self, frame: Esp32CsiFrame) -> None:
        now = frame.ts
        self.last_seen = now
        self.frames_total += 1
        self.frame_times_1s.append(now)
        self.rssi_dbm = frame.rssi_dbm
        self.n_subcarriers = frame.n_subcarriers
        self.n_antennas = frame.n_antennas
        self.channel_freq_mhz = frame.channel_freq_mhz
        if self.last_sequence is not None:
            self.seq_total += 1
            delta = (frame.sequence - self.last_sequence) & 0xFFFFFFFF
            if delta > 1:
                self.seq_gaps += int(delta - 1)
        self.last_sequence = frame.sequence
        self.window_frames.append(frame)

    def pps_1s(self, now: float) -> float:
        cutoff = now - 1.0
        while self.frame_times_1s and self.frame_times_1s[0] < cutoff:
            self.frame_times_1s.popleft()
        return float(len(self.frame_times_1s))

    def status(self, now: float) -> str:
        if self.last_seen <= 0:
            return "offline"
        age = now - self.last_seen
        if age < OK_SECS:
            return "ok"
        if age < STALE_SECS:
            return "stale"
        return "offline"

    def seq_gap_ratio(self) -> float:
        if self.seq_total <= 0:
            return 0.0
        return self.seq_gaps / self.seq_total


@dataclass
class Esp32WindowRow:
    ts_start: float
    ts_end: float
    node_id: int
    source_ip: str
    features: List[float]
    frames: int
    loss_ratio: float


class Esp32IngestService:
    def __init__(
        self,
        repo_root: Path,
        *,
        udp_port: int = 5005,
        window_seconds: float = 2.0,
        expected_fps: float = 20.0,
        num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS,
        on_window: Optional[Callable[[Esp32WindowRow], None]] = None,
    ) -> None:
        self.repo_root = repo_root
        self.udp_port = udp_port
        self.window_seconds = window_seconds
        self.expected_frames = max(1, int(window_seconds * expected_fps))
        self.num_subcarriers = num_subcarriers
        self.on_window = on_window
        self.store = Esp32StateStore(repo_root)
        self._nodes: Dict[str, _NodeTrack] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._udp_thread: Optional[threading.Thread] = None
        self._window_thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._csi_frames = 0
        self._sibling_skipped = 0
        self._parse_errors = 0
        self._windows_emitted = 0

    def start(self) -> None:
        if self._udp_thread is not None:
            return
        self._stop.clear()
        self.store.set_meta(
            running=True,
            udp_port=self.udp_port,
            window_seconds=self.window_seconds,
        )
        self._udp_thread = threading.Thread(target=self._udp_loop, name="esp32-udp", daemon=True)
        self._window_thread = threading.Thread(target=self._window_loop, name="esp32-window", daemon=True)
        self._udp_thread.start()
        self._window_thread.start()
        self.store.log(f"ingest started on UDP :{self.udp_port}")

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        for t in (self._udp_thread, self._window_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._udp_thread = None
        self._window_thread = None
        self.store.set_meta(
            running=False,
            udp_port=self.udp_port,
            window_seconds=self.window_seconds,
        )
        self._flush_state()
        self.store.log("ingest stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
                self._flush_state()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _udp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.udp_port))
        except OSError as exc:
            self.store.log(f"bind failed: {exc}")
            self._stop.set()
            return
        sock.settimeout(0.5)
        self._sock = sock
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_datagram(data, addr[0])

    def _handle_datagram(self, data: bytes, source_ip: str) -> None:
        try:
            frames, _ = parse_stream(data, source_ip=source_ip)
        except Exception:
            with self._lock:
                self._parse_errors += 1
            return

        if not frames:
            magic = int.from_bytes(data[:4], "little") if len(data) >= 4 else 0
            from mac_app.capture.esp32_parser import ruview_sibling_packet_name

            if ruview_sibling_packet_name(magic):
                with self._lock:
                    self._sibling_skipped += 1
            else:
                with self._lock:
                    self._parse_errors += 1
            return

        with self._lock:
            self._csi_frames += len(frames)
            for frame in frames:
                key = f"{source_ip}:{frame.node_id}"
                track = self._nodes.get(key)
                if track is None:
                    track = _NodeTrack(node_id=frame.node_id, source_ip=source_ip)
                    self._nodes[key] = track
                track.record_frame(frame)

    def _window_loop(self) -> None:
        next_tick = time.monotonic() + self.window_seconds
        while not self._stop.is_set():
            time.sleep(max(0.05, next_tick - time.monotonic()))
            if time.monotonic() < next_tick:
                continue
            next_tick += self.window_seconds
            self._emit_windows()

    def _emit_windows(self) -> None:
        now = time.time()
        rows: List[Esp32WindowRow] = []
        with self._lock:
            for track in self._nodes.values():
                frames = track.window_frames
                track.window_frames = []
                feats, _, loss = aggregate_esp32_window_features(
                    frames,
                    expected_frames=self.expected_frames,
                    num_subcarriers=self.num_subcarriers,
                )
                track.frames_last_window = len(frames)
                track.loss_ratio_last_window = loss
                if frames:
                    rows.append(
                        Esp32WindowRow(
                            ts_start=frames[0].ts,
                            ts_end=frames[-1].ts,
                            node_id=track.node_id,
                            source_ip=track.source_ip,
                            features=feats,
                            frames=len(frames),
                            loss_ratio=loss,
                        )
                    )
            self._windows_emitted += len(rows)
            fused = self._fuse_rows(rows)

        if fused is not None:
            self.store.update_fused_window(fused)

        for row in rows:
            if self.on_window is not None:
                self.on_window(row)

    @staticmethod
    def _fuse_rows(rows: List[Esp32WindowRow]) -> Optional[FusedWindowSnapshot]:
        if not rows:
            return None
        feats = np.mean([np.asarray(r.features, dtype=np.float32) for r in rows], axis=0)
        return FusedWindowSnapshot(
            window_id=str(uuid.uuid4()),
            ts_end=max(r.ts_end for r in rows),
            features=feats.tolist(),
            frames=sum(r.frames for r in rows),
            node_count=len(rows),
        )

    def _flush_state(self) -> None:
        now = time.time()
        with self._lock:
            nodes_out: List[NodeSnapshot] = []
            ok = stale = offline = 0
            total_pps = 0.0
            for track in self._nodes.values():
                st = track.status(now)
                if st == "ok":
                    ok += 1
                elif st == "stale":
                    stale += 1
                else:
                    offline += 1
                pps = track.pps_1s(now)
                total_pps += pps
                age = (now - track.last_seen) if track.last_seen > 0 else float("inf")
                nodes_out.append(
                    NodeSnapshot(
                        node_id=track.node_id,
                        source_ip=track.source_ip,
                        status=st,
                        last_seen_s_ago=age if age != float("inf") else -1.0,
                        frames_total=track.frames_total,
                        pps_1s=pps,
                        sequence=track.last_sequence or 0,
                        seq_gaps=track.seq_gaps,
                        seq_gap_ratio=track.seq_gap_ratio(),
                        rssi_dbm=track.rssi_dbm,
                        n_subcarriers=track.n_subcarriers,
                        n_antennas=track.n_antennas,
                        channel_freq_mhz=track.channel_freq_mhz,
                        frames_last_window=track.frames_last_window,
                        loss_ratio_last_window=track.loss_ratio_last_window,
                    )
                )
            agg = AggregateSnapshot(
                total_pps=total_pps,
                nodes_ok=ok,
                nodes_stale=stale,
                nodes_offline=offline,
                csi_frames_parsed=self._csi_frames,
                sibling_packets_skipped=self._sibling_skipped,
                parse_errors=self._parse_errors,
                windows_emitted=self._windows_emitted,
            )
        self.store.update_nodes(nodes_out, agg)
        self.store.flush()
