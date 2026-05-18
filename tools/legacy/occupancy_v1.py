"""Whole-home occupancy feature extraction and conservative scoring."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional


OCCUPANCY_LABELS = {
    "vacant",
    "occupied",
    "occupied_still",
    "occupied_moving",
    "unknown",
    "validation",
    "validation_vacant",
    "validation_occupied",
}
TRAINING_OCCUPANCY = {
    "vacant": "vacant",
    "occupied": "occupied",
    "occupied_still": "occupied",
    "occupied_moving": "occupied",
}
LABEL_OCCUPANCY = {
    **TRAINING_OCCUPANCY,
    "validation_vacant": "vacant",
    "validation_occupied": "occupied",
    "validation": "unknown",
    "unknown": "unknown",
}
MODEL_VERSION = 1

FEATURE_COLUMNS = [
    "window_id",
    "session_id",
    "phase",
    "timestamp_start",
    "timestamp_end",
    "scan_status",
    "scan_error",
    "sample_count",
    "visible_bssid_count",
    "rssi_avg_dbm",
    "rssi_min_dbm",
    "rssi_max_dbm",
    "rssi_std_db",
    "noise_avg_dbm",
    "snr_avg_db",
    "snr_min_db",
    "tx_bitrate_avg_mbps",
    "rx_bitrate_avg_mbps",
    "tx_mcs_mode",
    "rx_mcs_mode",
    "neighbor_count_same_channel",
    "neighbor_count_adjacent",
    "neighbor_rssi_sum_dbm",
    "channel_utilization_proxy",
    "missing_sample_count",
    "motion_score",
    "rssi_delta_prev_db",
    "snr_delta_prev_db",
    "rssi_recent_mean_dbm",
    "rssi_recent_std_db",
    "snr_recent_mean_db",
    "recent_motion_score",
]

LABEL_COLUMNS = [
    "block_id",
    "label",
    "occupancy_label",
    "is_training",
    "label_source",
    "source_detail",
    "label_confidence",
    "window_id",
    "session_id",
    "timestamp_start",
    "timestamp_end",
    "note",
]

STATE_COLUMNS = [
    "timestamp_start",
    "timestamp_end",
    "session_id",
    "window_id",
    "raw_state",
    "stable_state",
    "confidence",
    "occupied_score",
    "vacant_score",
    "reason",
    "model_id",
]

MODEL_FEATURES = [
    "visible_bssid_count",
    "rssi_avg_dbm",
    "rssi_std_db",
    "snr_avg_db",
    "channel_utilization_proxy",
    "missing_sample_count",
    "motion_score",
    "rssi_delta_prev_db",
    "rssi_recent_std_db",
    "recent_motion_score",
]

EVAL_ERROR_COLUMNS = [
    "window_id",
    "timestamp_start",
    "expected_state",
    "predicted_state",
    "raw_state",
    "confidence",
    "occupied_score",
    "vacant_score",
    "reason",
    "label",
    "label_source",
]


@dataclass
class OccupancyDecision:
    raw_state: str
    stable_state: str
    confidence: float
    occupied_score: Optional[float]
    vacant_score: Optional[float]
    reason: str


class OccupancyStateMachine:
    """Require repeated evidence before flipping stable occupied/vacant state."""

    def __init__(self, initial_state: str = "unknown", required_windows: int = 2):
        self.stable_state = initial_state
        self.required_windows = max(1, int(required_windows))
        self._candidate_state: Optional[str] = None
        self._candidate_count = 0

    def update(self, raw_state: str) -> str:
        if raw_state not in ("occupied", "vacant"):
            self._candidate_state = None
            self._candidate_count = 0
            return "unknown" if self.stable_state == "unknown" else self.stable_state

        if raw_state == self.stable_state:
            self._candidate_state = None
            self._candidate_count = 0
            return self.stable_state

        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        if self._candidate_count >= self.required_windows:
            self.stable_state = raw_state
            self._candidate_state = None
            self._candidate_count = 0
        return self.stable_state


def safe_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and value.lower() in ("none", "nan"):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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


def _rounded(value: Optional[float], digits: int = 3):
    return round(value, digits) if value is not None else ""


def extract_feature_row(window, history: list[dict], motion_score: Optional[float] = None) -> dict:
    summary = window.summary
    recent = history[-5:]
    prev = history[-1] if history else {}
    rssi = safe_float(summary.get("rssi_avg_dbm"))
    snr = safe_float(summary.get("snr_avg_db"))
    prev_rssi = safe_float(prev.get("rssi_avg_dbm"))
    prev_snr = safe_float(prev.get("snr_avg_db"))
    recent_rssi = [safe_float(row.get("rssi_avg_dbm")) for row in recent]
    recent_snr = [safe_float(row.get("snr_avg_db")) for row in recent]
    recent_motion = [safe_float(row.get("motion_score")) for row in recent]

    row = {
        "window_id": window.window_id,
        "session_id": summary.get("session_id", ""),
        "phase": window.phase,
        "timestamp_start": summary.get("timestamp_start", ""),
        "timestamp_end": summary.get("timestamp_end", ""),
        "scan_status": window.status,
        "scan_error": window.error_message,
        "sample_count": summary.get("sample_count", ""),
        "visible_bssid_count": window.visible_bssid_count,
        "rssi_avg_dbm": summary.get("rssi_avg_dbm", ""),
        "rssi_min_dbm": summary.get("rssi_min_dbm", ""),
        "rssi_max_dbm": summary.get("rssi_max_dbm", ""),
        "rssi_std_db": summary.get("rssi_std_db", ""),
        "noise_avg_dbm": summary.get("noise_avg_dbm", ""),
        "snr_avg_db": summary.get("snr_avg_db", ""),
        "snr_min_db": summary.get("snr_min_db", ""),
        "tx_bitrate_avg_mbps": summary.get("tx_bitrate_avg_mbps", ""),
        "rx_bitrate_avg_mbps": summary.get("rx_bitrate_avg_mbps", ""),
        "tx_mcs_mode": summary.get("tx_mcs_mode", ""),
        "rx_mcs_mode": summary.get("rx_mcs_mode", ""),
        "neighbor_count_same_channel": summary.get("neighbor_count_same_channel", ""),
        "neighbor_count_adjacent": summary.get("neighbor_count_adjacent", ""),
        "neighbor_rssi_sum_dbm": summary.get("neighbor_rssi_sum_dbm", ""),
        "channel_utilization_proxy": summary.get("channel_utilization_proxy", ""),
        "missing_sample_count": summary.get("missing_sample_count", ""),
        "motion_score": _rounded(motion_score),
        "rssi_delta_prev_db": _rounded(rssi - prev_rssi if rssi is not None and prev_rssi is not None else None),
        "snr_delta_prev_db": _rounded(snr - prev_snr if snr is not None and prev_snr is not None else None),
        "rssi_recent_mean_dbm": _rounded(_mean(recent_rssi)),
        "rssi_recent_std_db": _rounded(_std(recent_rssi)),
        "snr_recent_mean_db": _rounded(_mean(recent_snr)),
        "recent_motion_score": _rounded(_mean(recent_motion)),
    }
    return {column: row.get(column, "") for column in FEATURE_COLUMNS}


def label_row(
    label: str,
    block_id: str,
    feature_row: dict,
    note: str = "",
    *,
    label_source: str = "manual",
    source_detail: str = "",
    label_confidence: str = "",
) -> dict:
    if label not in OCCUPANCY_LABELS:
        raise ValueError(f"Unsupported occupancy label: {label}")
    occupancy_label = LABEL_OCCUPANCY.get(label, "unknown")
    is_training = "1" if label in TRAINING_OCCUPANCY else "0"
    return {
        "block_id": block_id,
        "label": label,
        "occupancy_label": occupancy_label,
        "is_training": is_training,
        "label_source": label_source,
        "source_detail": source_detail,
        "label_confidence": label_confidence,
        "window_id": feature_row.get("window_id", ""),
        "session_id": feature_row.get("session_id", ""),
        "timestamp_start": feature_row.get("timestamp_start", ""),
        "timestamp_end": feature_row.get("timestamp_end", ""),
        "note": note,
    }


def train_profile_model(feature_rows: list[dict], label_rows: list[dict], *, session_id: str) -> dict:
    labels_by_window = {
        row.get("window_id"): row.get("occupancy_label")
        for row in label_rows
        if row.get("is_training") in ("1", 1, True) and row.get("occupancy_label") in ("vacant", "occupied")
    }
    grouped = {"vacant": [], "occupied": []}
    for row in feature_rows:
        label = labels_by_window.get(row.get("window_id"))
        if label in grouped:
            grouped[label].append(row)

    profiles = {}
    for state, rows in grouped.items():
        metrics = {}
        for feature in MODEL_FEATURES:
            values = [safe_float(row.get(feature)) for row in rows]
            clean = [v for v in values if v is not None]
            metrics[feature] = {
                "mean": round(_mean(clean), 4) if clean else None,
                "std": round(max(_std(clean), 0.001), 4) if clean else None,
            }
        profiles[state] = {"window_count": len(rows), "metrics": metrics}

    model_id = f"{session_id}_occupancy_v{MODEL_VERSION}"
    return {
        "version": MODEL_VERSION,
        "model_id": model_id,
        "session_id": session_id,
        "model_type": "conservative_profile_threshold",
        "features": MODEL_FEATURES,
        "profiles": profiles,
        "thresholds": {
            "unknown_threshold": 0.62,
            "score_margin": 0.12,
            "max_distance": 4.0,
            "required_state_windows": 2,
        },
    }


def model_is_ready(model: dict) -> bool:
    profiles = model.get("profiles", {})
    return (
        profiles.get("vacant", {}).get("window_count", 0) >= 2
        and profiles.get("occupied", {}).get("window_count", 0) >= 2
    )


def _profile_distance(row: dict, profile: dict, features: list[str]) -> Optional[float]:
    distances = []
    metrics = profile.get("metrics", {})
    for feature in features:
        value = safe_float(row.get(feature))
        metric = metrics.get(feature, {})
        mean = safe_float(metric.get("mean"))
        std = safe_float(metric.get("std"))
        if value is None or mean is None or std is None:
            continue
        distances.append(abs(value - mean) / max(std, 0.5))
    if not distances:
        return None
    return sum(distances) / len(distances)


def score_occupancy(row: dict, model: dict, state_machine: Optional[OccupancyStateMachine] = None) -> OccupancyDecision:
    thresholds = model.get("thresholds", {})
    unknown_threshold = float(thresholds.get("unknown_threshold", 0.62))
    score_margin = float(thresholds.get("score_margin", 0.12))
    max_distance = float(thresholds.get("max_distance", 4.0))
    features = model.get("features") or MODEL_FEATURES

    if row.get("scan_status") == "failed":
        stable = state_machine.update("unknown") if state_machine else "unknown"
        return OccupancyDecision("unknown", stable, 0.0, None, None, "scan_failed")

    occupied_distance = _profile_distance(row, model.get("profiles", {}).get("occupied", {}), features)
    vacant_distance = _profile_distance(row, model.get("profiles", {}).get("vacant", {}), features)
    if occupied_distance is None or vacant_distance is None:
        stable = state_machine.update("unknown") if state_machine else "unknown"
        return OccupancyDecision("unknown", stable, 0.0, occupied_distance, vacant_distance, "model_missing_features")

    occupied_score = max(0.0, 1.0 - min(occupied_distance, max_distance) / max_distance)
    vacant_score = max(0.0, 1.0 - min(vacant_distance, max_distance) / max_distance)
    confidence = abs(occupied_score - vacant_score)

    if confidence < unknown_threshold or abs(occupied_score - vacant_score) < score_margin:
        raw_state = "unknown"
        reason = "low_confidence"
    else:
        raw_state = "occupied" if occupied_score > vacant_score else "vacant"
        reason = "profile_match"

    stable = state_machine.update(raw_state) if state_machine else raw_state
    return OccupancyDecision(
        raw_state=raw_state,
        stable_state=stable,
        confidence=round(confidence, 3),
        occupied_score=round(occupied_score, 3),
        vacant_score=round(vacant_score, 3),
        reason=reason,
    )


def state_row(feature_row: dict, decision: OccupancyDecision, model: dict) -> dict:
    return {
        "timestamp_start": feature_row.get("timestamp_start", ""),
        "timestamp_end": feature_row.get("timestamp_end", ""),
        "session_id": feature_row.get("session_id", ""),
        "window_id": feature_row.get("window_id", ""),
        "raw_state": decision.raw_state,
        "stable_state": decision.stable_state,
        "confidence": decision.confidence,
        "occupied_score": decision.occupied_score if decision.occupied_score is not None else "",
        "vacant_score": decision.vacant_score if decision.vacant_score is not None else "",
        "reason": decision.reason,
        "model_id": model.get("model_id", ""),
    }


def evaluate_profile_model(feature_rows: list[dict], label_rows: list[dict], model: dict, *, session_id: str) -> dict:
    """Score labeled windows and summarize occupied/vacant validation quality."""
    features_by_window = {row.get("window_id"): row for row in feature_rows if row.get("window_id")}
    confusion = {
        "vacant": {"vacant": 0, "occupied": 0, "unknown": 0},
        "occupied": {"vacant": 0, "occupied": 0, "unknown": 0},
    }
    evaluated = []
    errors = []

    for label in label_rows:
        expected = label.get("occupancy_label")
        if expected not in ("vacant", "occupied"):
            continue
        feature = features_by_window.get(label.get("window_id"))
        if not feature:
            continue
        decision = score_occupancy(feature, model)
        predicted = decision.raw_state if decision.raw_state in ("vacant", "occupied") else "unknown"
        confusion[expected][predicted] += 1
        row = {
            "window_id": feature.get("window_id", ""),
            "timestamp_start": feature.get("timestamp_start", ""),
            "expected_state": expected,
            "predicted_state": predicted,
            "raw_state": decision.raw_state,
            "confidence": decision.confidence,
            "occupied_score": decision.occupied_score if decision.occupied_score is not None else "",
            "vacant_score": decision.vacant_score if decision.vacant_score is not None else "",
            "reason": decision.reason,
            "label": label.get("label", ""),
            "label_source": label.get("label_source", ""),
        }
        evaluated.append(row)
        if predicted != expected:
            errors.append(row)

    total = len(evaluated)
    correct = sum(1 for row in evaluated if row["predicted_state"] == row["expected_state"])
    known = sum(1 for row in evaluated if row["predicted_state"] in ("vacant", "occupied"))
    false_positive_occupied = confusion["vacant"]["occupied"]
    false_negative_occupied = confusion["occupied"]["vacant"] + confusion["occupied"]["unknown"]

    return {
        "version": 1,
        "session_id": session_id,
        "model_id": model.get("model_id", ""),
        "model_type": model.get("model_type", ""),
        "thresholds": model.get("thresholds", {}),
        "total_labeled_windows": total,
        "known_prediction_windows": known,
        "correct_windows": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "known_rate": round(known / total, 4) if total else None,
        "confusion": confusion,
        "false_positive_occupied": false_positive_occupied,
        "false_negative_occupied": false_negative_occupied,
        "error_count": len(errors),
        "errors": errors,
    }
