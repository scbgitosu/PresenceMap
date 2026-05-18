"""Align YOLO frame observations with RF windows to produce GTLabels.

A frame falls in window W iff its timestamp lies in ``[W.ts_start, W.ts_end)``.
For each window we aggregate the YOLO frames assigned to it: ``person_count =
max(per-frame person count)``, ``label_confidence = mean(per-frame max conf)``.
If no frames fell in the window, the label is emitted with
``label_source = "missing"`` and downstream training excludes it.

Timestamps in the wire format are ISO-8601 strings (windows) and microseconds
since epoch (frames). The labeler converts everything to ``datetime`` objects
internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        return datetime.strptime(ts, ISO_FMT).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(ts)


@dataclass
class YoloObservation:
    frame_id: str
    ts_us: int
    person_count: int
    max_conf: float
    jpeg_path: Optional[str] = None

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.ts_us / 1_000_000, tz=timezone.utc)


@dataclass
class WindowKey:
    window_id: str
    session_id: str
    ts_start: str
    ts_end: str
    phase: str = "freeform"

    @property
    def start_dt(self) -> datetime:
        return _parse_iso(self.ts_start)

    @property
    def end_dt(self) -> datetime:
        return _parse_iso(self.ts_end)


@dataclass
class LabeledWindow:
    window: WindowKey
    person_count: int
    label_confidence: float
    label_source: str
    yolo_frame_refs: List[str] = field(default_factory=list)
    yolo_model_id: str = ""

    @property
    def occupancy(self) -> str:
        return "occupied" if self.person_count >= 1 else "vacant"

    def to_row(self) -> Dict[str, Any]:
        return {
            "window_id": self.window.window_id,
            "session_id": self.window.session_id,
            "ts_start": self.window.ts_start,
            "ts_end": self.window.ts_end,
            "occupancy": self.occupancy,
            "person_count": self.person_count,
            "label_confidence": self.label_confidence,
            "label_source": self.label_source,
            "yolo_frame_refs": list(self.yolo_frame_refs),
            "yolo_model_id": self.yolo_model_id,
            "note": "",
        }


def label_windows(
    windows: List[WindowKey],
    observations: List[YoloObservation],
    *,
    yolo_model_id: str,
    override: Optional[Dict[str, str]] = None,
) -> List[LabeledWindow]:
    """Align YOLO observations with RF windows and produce labels.

    ``override`` is a dict ``{window_id: forced_occupancy_string}`` so the
    Collect page can stamp manual phase blocks (labeled_vacant /
    labeled_occupied) over the YOLO output.
    """
    # Sort once.
    windows = sorted(windows, key=lambda w: w.start_dt)
    observations = sorted(observations, key=lambda o: o.ts_us)

    out: List[LabeledWindow] = []
    j = 0
    for w in windows:
        start, end = w.start_dt, w.end_dt
        # Advance observation pointer to first obs in this window.
        while j < len(observations) and observations[j].datetime_utc < start:
            j += 1
        bucket: List[YoloObservation] = []
        k = j
        while k < len(observations) and observations[k].datetime_utc < end:
            bucket.append(observations[k])
            k += 1
        forced = (override or {}).get(w.window_id)
        if forced is not None:
            person_count = 1 if forced == "occupied" else 0
            out.append(
                LabeledWindow(
                    window=w,
                    person_count=person_count,
                    label_confidence=1.0,
                    label_source="manual",
                    yolo_frame_refs=[o.frame_id for o in bucket],
                    yolo_model_id=yolo_model_id,
                )
            )
            continue
        if not bucket:
            out.append(
                LabeledWindow(
                    window=w,
                    person_count=0,
                    label_confidence=0.0,
                    label_source="missing",
                    yolo_frame_refs=[],
                    yolo_model_id=yolo_model_id,
                )
            )
            continue
        person_count = max(o.person_count for o in bucket)
        mean_conf = sum(o.max_conf for o in bucket) / len(bucket)
        out.append(
            LabeledWindow(
                window=w,
                person_count=person_count,
                label_confidence=mean_conf,
                label_source="yolo",
                yolo_frame_refs=[o.frame_id for o in bucket],
                yolo_model_id=yolo_model_id,
            )
        )
    return out
