"""Privacy-first webcam-derived occupancy labels.

The webcam is used only to produce ground-truth labels for RF training. The
deployed RF model must not consume image-derived features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


WEBCAM_CALIBRATION_VERSION = 1

WEBCAM_EVENT_COLUMNS = [
    "timestamp_start",
    "timestamp_end",
    "session_id",
    "window_id",
    "occupancy_label",
    "label",
    "confidence",
    "motion_ratio",
    "frame_count",
    "source_detail",
    "debug_artifact",
    "note",
]

WEBCAM_HEALTH_COLUMNS = [
    "timestamp",
    "session_id",
    "window_id",
    "status",
    "camera_index",
    "frames_captured",
    "brightness_avg",
    "motion_ratio",
    "error",
    "debug_artifact",
]


@dataclass
class WebcamObservation:
    status: str
    label: str
    occupancy_label: str
    confidence: float
    motion_ratio: Optional[float]
    frame_count: int
    brightness_avg: Optional[float]
    source_detail: str
    debug_artifact: str = ""
    note: str = ""


class WebcamOccupancyState:
    """Convert motion observations into conservative room occupancy labels."""

    def __init__(self, *, hold_seconds: float = 180.0):
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.last_motion_at: Optional[float] = None

    def derive(
        self,
        *,
        observed_at: float,
        motion_ratio: Optional[float],
        frame_count: int,
        brightness_avg: Optional[float],
        min_frames: int,
        motion_threshold: float,
        low_light_threshold: float,
        source_detail: str,
        status: str = "ok",
        error: str = "",
        debug_artifact: str = "",
    ) -> WebcamObservation:
        if status != "ok":
            return WebcamObservation(
                status=status,
                label="unknown",
                occupancy_label="unknown",
                confidence=0.0,
                motion_ratio=motion_ratio,
                frame_count=frame_count,
                brightness_avg=brightness_avg,
                source_detail=source_detail,
                debug_artifact=debug_artifact,
                note=error or status,
            )
        if frame_count < min_frames:
            return WebcamObservation(
                status="insufficient_frames",
                label="unknown",
                occupancy_label="unknown",
                confidence=0.0,
                motion_ratio=motion_ratio,
                frame_count=frame_count,
                brightness_avg=brightness_avg,
                source_detail=source_detail,
                debug_artifact=debug_artifact,
                note=f"captured {frame_count}/{min_frames} required frames",
            )
        if brightness_avg is not None and brightness_avg < low_light_threshold:
            return WebcamObservation(
                status="low_light",
                label="unknown",
                occupancy_label="unknown",
                confidence=0.0,
                motion_ratio=motion_ratio,
                frame_count=frame_count,
                brightness_avg=brightness_avg,
                source_detail=source_detail,
                debug_artifact=debug_artifact,
                note=f"brightness {brightness_avg:.1f} below threshold {low_light_threshold:.1f}",
            )

        ratio = motion_ratio or 0.0
        if ratio >= motion_threshold:
            self.last_motion_at = observed_at
            confidence = min(0.99, max(0.55, 0.55 + ratio * 8.0))
            return WebcamObservation(
                status="ok",
                label="occupied_moving",
                occupancy_label="occupied",
                confidence=round(confidence, 3),
                motion_ratio=ratio,
                frame_count=frame_count,
                brightness_avg=brightness_avg,
                source_detail=source_detail,
                debug_artifact=debug_artifact,
                note="motion observed; no identity inference",
            )

        if self.last_motion_at is not None and observed_at - self.last_motion_at <= self.hold_seconds:
            age = max(0.0, observed_at - self.last_motion_at)
            confidence = max(0.5, 0.8 - (age / max(self.hold_seconds, 1.0)) * 0.25)
            return WebcamObservation(
                status="ok",
                label="occupied_still",
                occupancy_label="occupied",
                confidence=round(confidence, 3),
                motion_ratio=ratio,
                frame_count=frame_count,
                brightness_avg=brightness_avg,
                source_detail=source_detail,
                debug_artifact=debug_artifact,
                note="recent motion hold; no identity inference",
            )

        return WebcamObservation(
            status="ok",
            label="vacant",
            occupancy_label="vacant",
            confidence=0.65,
            motion_ratio=ratio,
            frame_count=frame_count,
            brightness_avg=brightness_avg,
            source_detail=source_detail,
            debug_artifact=debug_artifact,
            note="no motion observed; verify with review for still occupants",
        )


def webcam_event_row(window, observation: WebcamObservation) -> dict:
    summary = window.summary
    return {
        "timestamp_start": summary.get("timestamp_start", ""),
        "timestamp_end": summary.get("timestamp_end", ""),
        "session_id": summary.get("session_id", ""),
        "window_id": window.window_id,
        "occupancy_label": observation.occupancy_label,
        "label": observation.label,
        "confidence": observation.confidence,
        "motion_ratio": observation.motion_ratio if observation.motion_ratio is not None else "",
        "frame_count": observation.frame_count,
        "source_detail": observation.source_detail,
        "debug_artifact": observation.debug_artifact,
        "note": observation.note,
    }


def webcam_health_row(*, timestamp: str, session_id: str, window_id: str, camera_index: int, observation: WebcamObservation) -> dict:
    return {
        "timestamp": timestamp,
        "session_id": session_id,
        "window_id": window_id,
        "status": observation.status,
        "camera_index": camera_index,
        "frames_captured": observation.frame_count,
        "brightness_avg": observation.brightness_avg if observation.brightness_avg is not None else "",
        "motion_ratio": observation.motion_ratio if observation.motion_ratio is not None else "",
        "error": observation.note if observation.status != "ok" else "",
        "debug_artifact": observation.debug_artifact,
    }


def summarize_calibration_samples(samples: list[dict]) -> dict:
    if not samples:
        return {
            "sample_count": 0,
            "brightness_mean": None,
            "brightness_min": None,
            "brightness_max": None,
            "motion_mean": None,
            "motion_max": None,
            "unknown_count": 0,
        }
    brightness = [float(row["brightness_avg"]) for row in samples if row.get("brightness_avg") not in (None, "")]
    motion = [float(row["motion_ratio"]) for row in samples if row.get("motion_ratio") not in (None, "")]
    return {
        "sample_count": len(samples),
        "brightness_mean": round(sum(brightness) / len(brightness), 3) if brightness else None,
        "brightness_min": round(min(brightness), 3) if brightness else None,
        "brightness_max": round(max(brightness), 3) if brightness else None,
        "motion_mean": round(sum(motion) / len(motion), 5) if motion else None,
        "motion_max": round(max(motion), 5) if motion else None,
        "unknown_count": sum(1 for row in samples if row.get("occupancy_label") == "unknown"),
    }


def recommend_motion_threshold(empty_summary: dict, person_summary: dict, fallback: float = 0.015) -> float:
    empty_max = empty_summary.get("motion_max")
    person_mean = person_summary.get("motion_mean")
    if empty_max is None and person_mean is None:
        return fallback
    if empty_max is None:
        return max(0.001, round(float(person_mean) * 0.5, 5))
    if person_mean is None:
        return max(fallback, round(float(empty_max) * 2.0, 5))
    return max(0.001, round((float(empty_max) + float(person_mean)) / 2.0, 5))
