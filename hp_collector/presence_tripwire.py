"""Headless continuous Wi-Fi tripwire collector for PresenceMap.

This module reuses hp_collector.wifi_scan for the actual RF observations and
adds a small baseline/event layer around it for HP-side unattended runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hp_collector.config_loader import load_project
from hp_collector.wifi_scan import classify_scan_batch, collect_samples, list_wifi_interfaces, summarize
from shared.csv_schema import RAW_COLUMNS
from shared.models import Sample
from shared.occupancy import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OCCUPANCY_LABELS,
    STATE_COLUMNS,
    OccupancyStateMachine,
    extract_feature_row,
    label_row,
    model_is_ready,
    score_occupancy,
    state_row,
    train_profile_model,
)
from shared.utils import now_iso

logger = logging.getLogger(__name__)

PRESENCE_RAW_COLUMNS = [
    "window_id",
    "phase",
    "window_started_at",
    "window_ended_at",
    "target_sample_count",
    "target_rssi_avg_dbm",
    "target_rssi_std_db",
    "target_snr_avg_db",
    "visible_bssid_count",
    "scan_status",
    "scan_error",
] + RAW_COLUMNS

PRESENCE_EVENT_COLUMNS = [
    "event_id",
    "timestamp_start",
    "timestamp_end",
    "session_id",
    "event_type",
    "confidence",
    "score",
    "rssi_delta_db",
    "rssi_std_delta_db",
    "snr_delta_db",
    "visible_bssid_delta",
    "channel_utilization_delta",
    "baseline_id",
    "window_id",
    "target_ssid",
    "target_bssid",
    "best_bssid",
    "interface",
    "scan_backend",
    "note",
]

BASELINE_VERSION = 1


@dataclass
class PresenceWindow:
    window_id: str
    phase: str
    samples: list[Sample]
    summary: dict
    visible_bssid_count: int
    status: str
    error_message: str


@dataclass
class TripwireEvent:
    event_type: str
    confidence: float
    score: float
    rssi_delta_db: Optional[float]
    rssi_std_delta_db: Optional[float]
    snr_delta_db: Optional[float]
    visible_bssid_delta: Optional[float]
    channel_utilization_delta: Optional[float]
    note: str = ""


def _open_writer(path: Path, columns: list[str]) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return handle, writer


def ensure_csv(path: Path, columns: list[str]) -> None:
    handle, _writer = _open_writer(path, columns)
    handle.close()


def _safe_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _std(values: Iterable[Optional[float]]) -> float:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    return statistics.stdev(clean)


def _visible_bssids(samples: list[Sample]) -> set[str]:
    return {s.bssid for s in samples if s.bssid and s.scan_backend != "error"}


def _target_context(session_id: str, window_id: str) -> dict:
    return {
        "click_id": window_id,
        "session_id": session_id,
        "router_position_id": "presence_tripwire",
        "x_px": 0,
        "y_px": 0,
        "room_id": "presence_zone",
        "room_name": "Presence zone",
        "waypoint_id": "",
        "height_ft": 0,
    }


def presence_session_dir(project_dir: Path, session_id: str) -> Path:
    return Path(project_dir) / "presence_sessions" / session_id


def default_baseline_path(session_dir: Path) -> Path:
    return session_dir / "presence_baseline.json"


def default_model_path(session_dir: Path) -> Path:
    return session_dir / "presence_model.json"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, columns: list[str], row: dict) -> None:
    handle, writer = _open_writer(path, columns)
    try:
        writer.writerow(row)
        handle.flush()
    finally:
        handle.close()


def _append_feature(features_path: Path, row: dict) -> None:
    _append_row(features_path, FEATURE_COLUMNS, row)


def _append_label(labels_path: Path, row: dict) -> None:
    _append_row(labels_path, LABEL_COLUMNS, row)


def _append_state(states_path: Path, row: dict) -> None:
    _append_row(states_path, STATE_COLUMNS, row)


def _first_window_number(history: list[dict], requested: int) -> int:
    if requested != 1:
        return requested
    return len([row for row in history if row.get("window_id")]) + 1


def collect_presence_window(
    *,
    interface: str,
    ssid: str,
    bssid: Optional[str],
    session_id: str,
    window_number: int,
    samples_per_window: int,
    delay_s: float,
    backend: str,
    phase: str,
) -> PresenceWindow:
    window_id = f"{session_id}_w{window_number:06d}"
    samples = collect_samples(
        interface=interface,
        ssid=ssid,
        bssid=bssid,
        samples=samples_per_window,
        delay_s=delay_s,
        click_context=_target_context(session_id, window_id),
        backend=backend,
    )
    summary = summarize(samples, ssid, bssid)
    outcome = classify_scan_batch(samples, ssid, bssid)
    return PresenceWindow(
        window_id=window_id,
        phase=phase,
        samples=samples,
        summary=summary,
        visible_bssid_count=len(_visible_bssids(samples)),
        status=outcome.status,
        error_message=outcome.error_message,
    )


def append_presence_raw(raw_path: Path, window: PresenceWindow) -> None:
    handle, writer = _open_writer(raw_path, PRESENCE_RAW_COLUMNS)
    try:
        for sample in window.samples:
            row = {
                "window_id": window.window_id,
                "phase": window.phase,
                "window_started_at": window.summary.get("timestamp_start", ""),
                "window_ended_at": window.summary.get("timestamp_end", ""),
                "target_sample_count": window.summary.get("sample_count"),
                "target_rssi_avg_dbm": window.summary.get("rssi_avg_dbm"),
                "target_rssi_std_db": window.summary.get("rssi_std_db"),
                "target_snr_avg_db": window.summary.get("snr_avg_db"),
                "visible_bssid_count": window.visible_bssid_count,
                "scan_status": window.status,
                "scan_error": window.error_message,
            }
            row.update(sample.to_dict())
            writer.writerow(row)
        handle.flush()
    finally:
        handle.close()


def build_baseline(
    *,
    windows: list[PresenceWindow],
    session_id: str,
    interface: str,
    ssid: str,
    bssid: Optional[str],
    backend: str,
    location_label: str,
) -> dict:
    usable = [w for w in windows if _safe_float(w.summary.get("rssi_avg_dbm")) is not None]
    if len(usable) < 2:
        raise RuntimeError("Need at least two usable calibration windows to build a baseline")

    rssi_values = [_safe_float(w.summary.get("rssi_avg_dbm")) for w in usable]
    rssi_std_values = [_safe_float(w.summary.get("rssi_std_db")) for w in usable]
    snr_values = [_safe_float(w.summary.get("snr_avg_db")) for w in usable]
    bssid_counts = [float(w.visible_bssid_count) for w in usable]
    channel_values = [_safe_float(w.summary.get("channel_utilization_proxy")) for w in usable]

    baseline_id = f"{session_id}_{int(time.time())}"
    return {
        "version": BASELINE_VERSION,
        "baseline_id": baseline_id,
        "created_at": now_iso(),
        "session_id": session_id,
        "location_label": location_label,
        "interface": interface,
        "scan_backend": backend,
        "target_ssid": ssid,
        "target_bssid": bssid or "",
        "window_count": len(usable),
        "discarded_window_count": len(windows) - len(usable),
        "metrics": {
            "rssi_avg_dbm": {
                "mean": round(_mean(rssi_values), 3),
                "std": round(_std(rssi_values), 3),
            },
            "rssi_std_db": {
                "mean": round(_mean(rssi_std_values), 3),
                "std": round(_std(rssi_std_values), 3),
            },
            "snr_avg_db": {
                "mean": round(_mean(snr_values), 3) if _mean(snr_values) is not None else None,
                "std": round(_std(snr_values), 3),
            },
            "visible_bssid_count": {
                "mean": round(_mean(bssid_counts), 3),
                "std": round(_std(bssid_counts), 3),
            },
            "channel_utilization_proxy": {
                "mean": round(_mean(channel_values), 3) if _mean(channel_values) is not None else None,
                "std": round(_std(channel_values), 3),
            },
        },
    }


def save_baseline(path: Path, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_model(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_model(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    if baseline.get("version") != BASELINE_VERSION:
        raise RuntimeError(f"Unsupported baseline version in {path}")
    return baseline


def _zscore_delta(value: Optional[float], metric: dict, floor: float) -> tuple[float, Optional[float]]:
    if value is None or metric.get("mean") is None:
        return 0.0, None
    delta = value - float(metric["mean"])
    scale = max(float(metric.get("std") or 0.0), floor)
    return abs(delta) / scale, delta


def score_window(window: PresenceWindow, baseline: dict, threshold: float) -> Optional[TripwireEvent]:
    metrics = baseline["metrics"]
    rssi_score, rssi_delta = _zscore_delta(
        _safe_float(window.summary.get("rssi_avg_dbm")),
        metrics["rssi_avg_dbm"],
        floor=2.0,
    )
    rssi_std_score, rssi_std_delta = _zscore_delta(
        _safe_float(window.summary.get("rssi_std_db")),
        metrics["rssi_std_db"],
        floor=1.0,
    )
    snr_score, snr_delta = _zscore_delta(
        _safe_float(window.summary.get("snr_avg_db")),
        metrics["snr_avg_db"],
        floor=3.0,
    )
    bssid_score, bssid_delta = _zscore_delta(
        float(window.visible_bssid_count),
        metrics["visible_bssid_count"],
        floor=2.0,
    )
    channel_score, channel_delta = _zscore_delta(
        _safe_float(window.summary.get("channel_utilization_proxy")),
        metrics["channel_utilization_proxy"],
        floor=8.0,
    )

    score = max(rssi_score, rssi_std_score * 0.8, snr_score * 0.6, bssid_score * 0.4, channel_score * 0.4)
    if window.status == "failed":
        return None
    if score < threshold:
        return None

    confidence = min(0.99, 0.45 + (score - threshold) / max(threshold * 2.0, 1.0))
    note_parts = []
    if rssi_delta is not None:
        note_parts.append(f"rssi_delta={rssi_delta:+.2f}dB")
    if rssi_std_delta is not None:
        note_parts.append(f"rssi_std_delta={rssi_std_delta:+.2f}dB")
    if snr_delta is not None:
        note_parts.append(f"snr_delta={snr_delta:+.2f}dB")

    return TripwireEvent(
        event_type="motion",
        confidence=round(confidence, 3),
        score=round(score, 3),
        rssi_delta_db=round(rssi_delta, 3) if rssi_delta is not None else None,
        rssi_std_delta_db=round(rssi_std_delta, 3) if rssi_std_delta is not None else None,
        snr_delta_db=round(snr_delta, 3) if snr_delta is not None else None,
        visible_bssid_delta=round(bssid_delta, 3) if bssid_delta is not None else None,
        channel_utilization_delta=round(channel_delta, 3) if channel_delta is not None else None,
        note="; ".join(note_parts),
    )


def append_event(events_path: Path, window: PresenceWindow, event: TripwireEvent, baseline: dict) -> None:
    handle, writer = _open_writer(events_path, PRESENCE_EVENT_COLUMNS)
    try:
        writer.writerow({
            "event_id": f"{window.window_id}_motion",
            "timestamp_start": window.summary.get("timestamp_start", ""),
            "timestamp_end": window.summary.get("timestamp_end", ""),
            "session_id": baseline.get("session_id", ""),
            "event_type": event.event_type,
            "confidence": event.confidence,
            "score": event.score,
            "rssi_delta_db": event.rssi_delta_db,
            "rssi_std_delta_db": event.rssi_std_delta_db,
            "snr_delta_db": event.snr_delta_db,
            "visible_bssid_delta": event.visible_bssid_delta,
            "channel_utilization_delta": event.channel_utilization_delta,
            "baseline_id": baseline.get("baseline_id", ""),
            "window_id": window.window_id,
            "target_ssid": baseline.get("target_ssid", ""),
            "target_bssid": baseline.get("target_bssid", ""),
            "best_bssid": window.summary.get("best_bssid", ""),
            "interface": baseline.get("interface", ""),
            "scan_backend": baseline.get("scan_backend", ""),
            "note": event.note,
        })
        handle.flush()
    finally:
        handle.close()


def run_calibration(args, config) -> Path:
    session_dir = presence_session_dir(args.project, args.session)
    raw_path = session_dir / "presence_raw.csv"
    events_path = session_dir / "presence_events.csv"
    baseline_path = Path(args.baseline_path) if args.baseline_path else default_baseline_path(session_dir)
    windows_needed = max(2, math.ceil(args.baseline_seconds / args.window_seconds))
    ensure_csv(raw_path, PRESENCE_RAW_COLUMNS)
    ensure_csv(events_path, PRESENCE_EVENT_COLUMNS)

    logger.info("calibrating %s windows into %s", windows_needed, baseline_path)
    windows: list[PresenceWindow] = []
    for window_number in range(1, windows_needed + 1):
        window_started = time.monotonic()
        window = collect_presence_window(
            interface=args.interface,
            ssid=args.ssid,
            bssid=args.bssid,
            session_id=args.session,
            window_number=window_number,
            samples_per_window=args.samples_per_window,
            delay_s=args.delay,
            backend=args.backend,
            phase="calibration",
        )
        append_presence_raw(raw_path, window)
        windows.append(window)
        logger.info(
            "calibration window %s/%s status=%s rssi=%s std=%s bssids=%s",
            window_number,
            windows_needed,
            window.status,
            window.summary.get("rssi_avg_dbm"),
            window.summary.get("rssi_std_db"),
            window.visible_bssid_count,
        )
        _sleep_until_next_window(window_started, args.window_seconds)

    baseline = build_baseline(
        windows=windows,
        session_id=args.session,
        interface=args.interface,
        ssid=args.ssid,
        bssid=args.bssid,
        backend=args.backend,
        location_label=args.location_label,
    )
    save_baseline(baseline_path, baseline)
    logger.info("saved baseline %s with %s usable windows", baseline["baseline_id"], baseline["window_count"])
    return baseline_path


def run_label_block(args, config) -> Path:
    session_dir = presence_session_dir(args.project, args.session)
    raw_path = session_dir / "presence_raw.csv"
    features_path = session_dir / "presence_features.csv"
    labels_path = session_dir / "presence_labels.csv"
    model_path = Path(args.occupancy_model_path) if args.occupancy_model_path else default_model_path(session_dir)
    windows_needed = max(1, math.ceil(args.block_seconds / args.window_seconds))
    block_id = f"{args.session}_{args.label_block}_{int(time.time())}"
    ensure_csv(raw_path, PRESENCE_RAW_COLUMNS)
    ensure_csv(features_path, FEATURE_COLUMNS)
    ensure_csv(labels_path, LABEL_COLUMNS)

    logger.info("collecting %s labeled %s windows into %s", windows_needed, args.label_block, session_dir)
    history = _read_csv_rows(features_path)
    start_window = _first_window_number(history, args.start_window)
    for window_number in range(start_window, start_window + windows_needed):
        window_started = time.monotonic()
        window = collect_presence_window(
            interface=args.interface,
            ssid=args.ssid,
            bssid=args.bssid,
            session_id=args.session,
            window_number=window_number,
            samples_per_window=args.samples_per_window,
            delay_s=args.delay,
            backend=args.backend,
            phase=f"label_{args.label_block}",
        )
        append_presence_raw(raw_path, window)
        motion_event = None
        feature = extract_feature_row(window, history, motion_score=motion_event.score if motion_event else None)
        _append_feature(features_path, feature)
        _append_label(labels_path, label_row(args.label_block, block_id, feature, note=args.note))
        history.append(feature)
        logger.info(
            "label=%s window=%s status=%s rssi=%s bssids=%s",
            args.label_block,
            window.window_id,
            window.status,
            window.summary.get("rssi_avg_dbm"),
            window.visible_bssid_count,
        )
        _sleep_until_next_window(window_started, args.window_seconds)

    feature_rows = _read_csv_rows(features_path)
    label_rows = _read_csv_rows(labels_path)
    model = train_profile_model(feature_rows, label_rows, session_id=args.session)
    save_model(model_path, model)
    if model_is_ready(model):
        logger.info("saved occupancy model %s to %s", model.get("model_id"), model_path)
    else:
        logger.info("saved partial occupancy model to %s; collect vacant and occupied blocks before live scoring", model_path)
    return model_path


def _sleep_until_next_window(window_started: float, window_seconds: float) -> None:
    elapsed = time.monotonic() - window_started
    if elapsed < window_seconds:
        time.sleep(max(0.0, window_seconds - elapsed))


def run_monitor(args, config) -> None:
    session_dir = presence_session_dir(args.project, args.session)
    raw_path = session_dir / "presence_raw.csv"
    events_path = session_dir / "presence_events.csv"
    features_path = session_dir / "presence_features.csv"
    states_path = session_dir / "presence_states.csv"
    baseline_path = Path(args.baseline_path) if args.baseline_path else default_baseline_path(session_dir)
    baseline = load_baseline(baseline_path) if baseline_path.exists() else None
    if baseline is None and not args.occupancy_monitor:
        baseline = load_baseline(baseline_path)
    occupancy_model = None
    state_machine = None
    if args.occupancy_monitor:
        model_path = Path(args.occupancy_model_path) if args.occupancy_model_path else default_model_path(session_dir)
        occupancy_model = load_model(model_path)
        if args.unknown_threshold is not None:
            occupancy_model.setdefault("thresholds", {})["unknown_threshold"] = args.unknown_threshold
        required = occupancy_model.get("thresholds", {}).get("required_state_windows", 2)
        state_machine = OccupancyStateMachine(required_windows=required)
        if not model_is_ready(occupancy_model):
            logger.warning("occupancy model has limited labels; live states may remain unknown")
    ensure_csv(raw_path, PRESENCE_RAW_COLUMNS)
    ensure_csv(events_path, PRESENCE_EVENT_COLUMNS)
    ensure_csv(features_path, FEATURE_COLUMNS)
    if args.occupancy_monitor:
        ensure_csv(states_path, STATE_COLUMNS)

    logger.info("monitoring with baseline %s", baseline.get("baseline_id") if baseline else "none")
    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("stop requested; finishing current window")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    last_event_at = 0.0
    last_state_at = 0.0
    window_number = args.start_window
    history = _read_csv_rows(features_path)
    window_number = _first_window_number(history, args.start_window)
    stop_after_window = window_number + args.max_windows if args.max_windows else None
    while not stop_requested:
        window_started = time.monotonic()
        window = collect_presence_window(
            interface=args.interface,
            ssid=args.ssid,
            bssid=args.bssid,
            session_id=args.session,
            window_number=window_number,
            samples_per_window=args.samples_per_window,
            delay_s=args.delay,
            backend=args.backend,
            phase="monitor",
        )
        append_presence_raw(raw_path, window)
        event = score_window(window, baseline, args.threshold) if baseline else None
        feature = extract_feature_row(window, history, motion_score=event.score if event else None)
        _append_feature(features_path, feature)
        history.append(feature)
        now = time.time()
        if event and now - last_event_at >= args.cooldown_seconds:
            append_event(events_path, window, event, baseline)
            last_event_at = now
            logger.info("event=%s confidence=%.3f score=%.3f %s", event.event_type, event.confidence, event.score, event.note)
        else:
            logger.info(
                "window=%s status=%s rssi=%s score=%s",
                window.window_id,
                window.status,
                window.summary.get("rssi_avg_dbm"),
                event.score if event else "quiet",
            )
        if occupancy_model and state_machine:
            decision = score_occupancy(feature, occupancy_model, state_machine)
            if now - last_state_at >= args.state_cooldown_seconds or decision.raw_state == "unknown":
                _append_state(states_path, state_row(feature, decision, occupancy_model))
                last_state_at = now
            logger.info(
                "occupancy raw=%s stable=%s confidence=%.3f reason=%s",
                decision.raw_state,
                decision.stable_state,
                decision.confidence,
                decision.reason,
            )
        window_number += 1
        if stop_after_window and window_number >= stop_after_window:
            break
        _sleep_until_next_window(window_started, args.window_seconds)


def _resolve_interface(config, requested: Optional[str]) -> str:
    if requested:
        return requested
    if config.default_interface:
        return config.default_interface
    ifaces = list_wifi_interfaces()
    if ifaces:
        return ifaces[0]
    raise RuntimeError("No Wi-Fi interface found; pass --interface wlan1")


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Headless PresenceMap Wi-Fi tripwire collector")
    parser.add_argument("--project", type=Path, required=True, help="Project directory under survey_projects/")
    parser.add_argument("--session", default="presence_tripwire", help="Presence session folder name")
    parser.add_argument("--interface", default=None, help="Wi-Fi interface, e.g. wlan1")
    parser.add_argument("--ssid", default=None, help="Target SSID; defaults to project_config.json")
    parser.add_argument("--bssid", default=None, help="Optional target BSSID; defaults to project_config.json")
    parser.add_argument("--backend", choices=["auto", "iw", "nmcli"], default=None)
    parser.add_argument("--samples-per-window", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between scans inside a window")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Approximate cadence between windows")
    parser.add_argument("--baseline-seconds", type=float, default=120.0)
    parser.add_argument("--baseline-path", default=None)
    parser.add_argument("--location-label", default="tripwire", help="Human label for this calibrated RF path")
    parser.add_argument("--threshold", type=float, default=2.5, help="Motion event threshold in baseline-normalized units")
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--state-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--unknown-threshold", type=float, default=None, help="Override occupancy unknown threshold")
    parser.add_argument("--start-window", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N monitor windows; 0 means run until interrupted")
    parser.add_argument("--calibrate", action="store_true", help="Collect a vacant baseline before monitoring")
    parser.add_argument("--monitor", action="store_true", help="Run continuous tripwire monitoring")
    parser.add_argument("--label-block", choices=sorted(OCCUPANCY_LABELS), default=None)
    parser.add_argument("--block-seconds", type=float, default=120.0)
    parser.add_argument("--occupancy-monitor", action="store_true", help="Run live conservative occupancy scoring")
    parser.add_argument("--occupancy-model-path", default=None)
    parser.add_argument("--note", default="", help="Optional note for labeled occupancy windows")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config, _rooms, _routers, _metadata = load_project(args.project)
    args.interface = _resolve_interface(config, args.interface)
    args.ssid = args.ssid or config.target_ssid
    args.bssid = args.bssid if args.bssid is not None else (config.target_bssid or None)
    args.backend = args.backend or config.scan_backend or "iw"

    if not args.ssid:
        raise RuntimeError("No target SSID configured; pass --ssid or set project_config.json")
    if args.label_block:
        args.monitor = False
    if not args.calibrate and not args.monitor and not args.label_block:
        args.monitor = True

    logger.info(
        "PresenceMap tripwire project=%s session=%s interface=%s ssid=%s backend=%s",
        args.project,
        args.session,
        args.interface,
        args.ssid,
        args.backend,
    )

    if args.calibrate:
        run_calibration(args, config)
    if args.label_block:
        run_label_block(args, config)
    if args.monitor:
        run_monitor(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
