"""Main per-window loop.

Stage 2 behavior: every ``window_seconds`` (default 2 s) run one ``iw scan``,
aggregate per-BSS signals into an ``RFSummary``, emit a WindowMsg and a
HealthMsg. CSI fields are left empty placeholders until Stage 3.

This loop is intentionally synchronous and single-threaded. The webcam
producer runs in its own thread (see ``hp_agent.runners.frame_loop``).
"""
from __future__ import annotations

import subprocess
import threading
from typing import Optional

from hp_agent.capture.csi_features import aggregate_window_features
from hp_agent.capture.rssi_iw import (
    RFSummary,
    read_noise_floor_dbm,
    run_iw_scan,
    summarize_window,
)
from hp_agent.config import AgentConfig
from hp_agent.health.heartbeat import HeartbeatEmitter
from hp_agent.runners.csi_buffer import CSIBuffer
from hp_agent.transport.messages import make_window_msg
from hp_agent.transport.zmq_publisher import Publisher
from hp_agent.util.logging import get_logger, log_event
from hp_agent.util.timing import sleep_until, window_deadlines
from shared.utils import now_iso


class WindowLoop:
    def __init__(
        self,
        cfg: AgentConfig,
        publisher: Publisher,
        heartbeat: HeartbeatEmitter,
        *,
        session_id: str,
        phase: str = "freeform",
        csi_buffer: Optional[CSIBuffer] = None,
    ) -> None:
        self.cfg = cfg
        self.publisher = publisher
        self.heartbeat = heartbeat
        self.session_id = session_id
        self.phase = phase
        self.csi_buffer = csi_buffer
        self.stop_event = threading.Event()
        log_path = cfg.paths.get("sessions_dir", cfg.project_dir) / session_id / "agent.log.jsonl"
        self.log = get_logger("hp_agent.window_loop", session_log=log_path)
        self._n = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        scan_period = max(self.cfg.window_seconds, 0.5)
        log_event(
            self.log,
            "INFO",
            "window_loop start",
            session_id=self.session_id,
            phase=self.phase,
            interface=self.cfg.interface,
            window_seconds=scan_period,
        )
        for deadline in window_deadlines(scan_period):
            if self.stop_event.is_set():
                break
            overrun = sleep_until(deadline)
            ts_start = now_iso()
            scan_error: Optional[str] = None
            summary: Optional[RFSummary] = None
            try:
                rows = run_iw_scan(self.cfg.interface, timeout_s=min(scan_period * 2, 6.0))
            except subprocess.CalledProcessError as exc:
                scan_error = f"iw scan exit {exc.returncode}: {(exc.stderr or '').strip()[:160]}"
                rows = []
            except subprocess.TimeoutExpired:
                scan_error = "iw scan timed out"
                rows = []
            except FileNotFoundError:
                scan_error = "iw binary missing"
                rows = []
            noise = read_noise_floor_dbm(self.cfg.interface)
            summary = summarize_window(
                rows,
                target_ssid=self.cfg.target_ssid,
                target_bssid=self.cfg.target_bssid,
                noise_dbm=noise,
            )
            ts_end = now_iso()
            self._n += 1
            window_id = f"{self.session_id}_w{self._n:06d}"
            rssi_payload = {
                "target_rssi_avg_dbm": summary.target_rssi_avg_dbm,
                "target_rssi_std_db": summary.target_rssi_std_db,
                "snr_db": summary.snr_db,
                "noise_dbm": summary.noise_dbm,
                "visible_bssid_count": summary.visible_bssid_count,
                "neighbor_rssi_sum_dbm": summary.neighbor_rssi_sum_dbm,
                "channel_utilization_proxy": summary.channel_utilization_proxy,
                "target_seen": summary.target_seen,
            }
            csi_payload = self._build_csi_payload()
            msg = make_window_msg(
                agent_id=self.cfg.agent_id,
                session_id=self.session_id,
                window_id=window_id,
                ts_start=ts_start,
                ts_end=ts_end,
                phase=self.phase,
                rssi=rssi_payload,
                csi=csi_payload,
                interface=self.cfg.interface,
                backend="iw_scan+csi" if self.csi_buffer is not None else "iw_scan",
            )
            self.publisher.send_window(msg)
            self.heartbeat.record_window(
                target_seen=summary.target_seen,
                loss_ratio=csi_payload["loss_ratio"] if csi_payload else (1.0 if scan_error else 0.0),
                overrun_s=overrun,
                scan_error=scan_error,
            )
            if scan_error:
                log_event(self.log, "WARNING", "scan failed", error=scan_error)
            if self.stop_event.is_set():
                break
        log_event(self.log, "INFO", "window_loop stop", session_id=self.session_id, windows=self._n)

    def _build_csi_payload(self) -> Optional[dict]:
        if self.csi_buffer is None:
            return None
        frames = self.csi_buffer.drain()
        include_amp = (
            bool(self.cfg.csi_include_amp_matrix) and self.phase != "live"
        )
        feats, amp_blob, loss = aggregate_window_features(
            frames,
            expected_frames=self.cfg.csi_expected_frames_per_window,
            num_subcarriers=self.cfg.csi_num_subcarriers,
            include_amp_matrix=include_amp,
        )
        payload: dict = {
            "features": feats,
            "frames": len(frames),
            "loss_ratio": loss,
        }
        if amp_blob is not None:
            payload["amp_matrix_f16"] = amp_blob
        return payload
