"""Evaluation metrics + error analysis for ``TemporalCSIModel``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from mac_app.train.features import TrainingDataset


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: List[List[int]]               # 2x2 [[tn, fp], [fn, tp]]
    per_window: List[Dict[str, object]] = field(default_factory=list)
    errors: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion": self.confusion,
            "per_window": self.per_window,
            "errors": self.errors,
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def evaluate(model, ds: TrainingDataset, *, device: str = "cpu") -> EvaluationResult:
    model.eval()
    X = torch.from_numpy(ds.X.astype(np.float32)).to(device)
    y = ds.y.astype(np.int64)
    with torch.no_grad():
        logits, motion = model(X)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    motion_np = motion.detach().cpu().numpy()
    tn = int(((preds == 0) & (y == 0)).sum())
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    n = len(y)
    accuracy = (tp + tn) / max(n, 1)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    per_window = []
    errors = []
    for i in range(n):
        row = {
            "window_id": str(ds.window_ids[i]),
            "session_id": str(ds.session_ids[i]),
            "y_true": int(y[i]),
            "y_pred": int(preds[i]),
            "p_occupied": float(probs[i, 1]),
            "motion_pred": float(motion_np[i]),
        }
        per_window.append(row)
        if int(preds[i]) != int(y[i]):
            errors.append(row)
    return EvaluationResult(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=[[tn, fp], [fn, tp]],
        per_window=per_window,
        errors=errors,
    )
