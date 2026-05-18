"""Model registry: persist + load TemporalCSIModel artifacts.

Layout: ``data/models/<model_id>/{weights.pt, meta.json, eval.json}``.
``model_id`` is ``<YYYYMMDDhhmmss>_<git_short_sha>_<sessions_hint>``.

``load_model`` rebuilds the model, the scaler, and the feature ordering --
the live inference path (Stage 6) uses this single helper.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from mac_app.train.eval import EvaluationResult
from mac_app.train.features import DatasetSpec, FeatureExtractor, feature_names
from mac_app.train.model import ModelConfig, TemporalCSIModel
from shared.project import models_dir
from shared.versioning import MODEL_VERSION, SCHEMA_VERSION


@dataclass
class ModelMeta:
    model_id: str
    model_dir: Path
    created_at: str
    session_ids: List[str]
    feature_names: List[str]
    model_cfg: dict
    train_cfg: dict
    eval_summary: dict
    n_params: int
    code_git_sha: str = ""
    schema_version: int = SCHEMA_VERSION
    model_version: int = MODEL_VERSION

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["model_dir"] = str(self.model_dir)
        return d


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=1.0)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _make_model_id(session_ids: List[str], created_at: str) -> str:
    import re
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()
    ts = dt.strftime("%Y%m%d%H%M%S")
    hint = "_".join(s for s in session_ids[:3]) or "session"
    hint = re.sub(r"[^A-Za-z0-9_-]", "", hint)[:32]
    sha = _git_sha() or "nogit"
    return f"{ts}_{sha}_{hint}"


def _project_root_for_models(project_dir: Path | str) -> Path:
    """Models live at ``<repo_root>/data/models/`` regardless of project_dir."""
    p = Path(project_dir).resolve()
    # Walk up until we find ``data/`` sibling.
    for candidate in [p, *p.parents]:
        if (candidate / "data").exists():
            return candidate
        if (candidate / "data" / "survey_projects").exists():
            return candidate
    return p


def save_model(
    *,
    project_dir: Path | str,
    model: TemporalCSIModel,
    extractor: FeatureExtractor,
    cfg: ModelConfig,
    train_cfg,
    session_ids: List[str],
    history: dict,
    eval_result: EvaluationResult,
    created_at: str,
    n_params: int,
) -> ModelMeta:
    root = _project_root_for_models(project_dir)
    md = models_dir(root)
    md.mkdir(parents=True, exist_ok=True)
    model_id = _make_model_id(session_ids, created_at)
    out_dir = md / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Weights + scaler in a single torch save so reload is one call.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": asdict(cfg),
            "scaler_mean": extractor.scaler.mean_.astype(np.float32),
            "scaler_scale": extractor.scaler.scale_.astype(np.float32),
            "feature_names": list(extractor.feature_names),
            "spec": asdict(extractor.spec),
        },
        out_dir / "weights.pt",
    )

    meta = ModelMeta(
        model_id=model_id,
        model_dir=out_dir,
        created_at=created_at,
        session_ids=list(session_ids),
        feature_names=list(extractor.feature_names),
        model_cfg=asdict(cfg),
        train_cfg=asdict(train_cfg),
        eval_summary={
            "accuracy": eval_result.accuracy,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "f1": eval_result.f1,
            "confusion": eval_result.confusion,
            "n_samples": len(eval_result.per_window),
            "n_errors": len(eval_result.errors),
        },
        n_params=int(n_params),
        code_git_sha=_git_sha(),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2, default=str))
    (out_dir / "eval.json").write_text(json.dumps(
        {
            "summary": meta.eval_summary,
            "history": history,
            "per_window": eval_result.per_window,
            "errors": eval_result.errors,
        },
        indent=2,
        default=str,
    ))
    return meta


def list_models(project_dir: Path | str) -> List[ModelMeta]:
    root = _project_root_for_models(project_dir)
    md = models_dir(root)
    if not md.exists():
        return []
    out: List[ModelMeta] = []
    for d in sorted(md.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        payload["model_dir"] = d
        # Coerce list-of-lists confusion back through ModelMeta.
        out.append(ModelMeta(**{k: payload[k] for k in ModelMeta.__dataclass_fields__ if k in payload}))
    return out


def load_model(
    project_dir: Path | str,
    model_id: str,
    *,
    device: str = "cpu",
) -> Tuple[TemporalCSIModel, FeatureExtractor]:
    root = _project_root_for_models(project_dir)
    md = models_dir(root) / model_id
    if not md.exists():
        raise FileNotFoundError(md)
    payload = torch.load(md / "weights.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**payload["model_cfg"])
    model = TemporalCSIModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    spec = DatasetSpec(**payload["spec"])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.with_mean = True
    scaler.with_std = True
    extractor = FeatureExtractor(
        spec=spec,
        feature_names=list(payload["feature_names"]),
        scaler=scaler,
    )
    return model, extractor


def read_eval(project_dir: Path | str, model_id: str) -> Optional[dict]:
    root = _project_root_for_models(project_dir)
    p = models_dir(root) / model_id / "eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
