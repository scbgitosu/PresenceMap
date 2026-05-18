"""YOLO person detector wrapper.

Uses ``ultralytics`` with PyTorch on Apple's MPS backend (or CPU if MPS is
unavailable). Loaded lazily so importing the module on a Mac without the
weights cached doesn't trigger a download.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "conf": self.conf,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


def _pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    """Single-class wrapper around ``ultralytics.YOLO``. Detects people only."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or _pick_device()
        self.verbose = verbose
        self._model = None  # lazy

    @property
    def model_id(self) -> str:
        return f"{self.model_path}::{self.device}::conf{self.conf_threshold}"

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # imported here so test envs without it still pass

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a BGR numpy array (as returned by ``cv2.imdecode``)."""
        model = self._load()
        results = model.predict(
            source=frame_bgr,
            classes=[0],   # COCO person class
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=self.verbose,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            out.append(
                Detection(
                    cls=int(k),
                    conf=float(c),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return out


def summarize(detections: List[Detection]) -> Tuple[int, float]:
    """Returns (person_count, max_confidence)."""
    if not detections:
        return 0, 0.0
    return len(detections), max(d.conf for d in detections)
