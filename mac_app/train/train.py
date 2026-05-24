"""Train ``TemporalCSIModel`` on one or more recorded sessions.

Use:
    >>> from mac_app.train.train import train_session
    >>> meta = train_session(project_dir, ["train001"])

For CLI use see ``mac_app/cli.py`` (``presence-mac train``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from mac_app.train.dataset import list_sessions
from mac_app.train.eval import EvaluationResult, evaluate
from mac_app.train.features import (
    DatasetSpec,
    TrainingDataset,
    build_dataset,
)
from mac_app.train.model import ModelConfig, TemporalCSIModel, count_parameters
from mac_app.train.registry import ModelMeta, save_model
from shared.utils import now_iso


def _pick_device(force: Optional[str] = None) -> str:
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, motion: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.motion = torch.from_numpy(motion.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.motion[idx]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.3
    motion_weight: float = 0.2     # weight of the motion-regression aux loss
    seed: int = 17
    device: Optional[str] = None   # default: mps if available


def _split_indices(n: int, val_fraction: float, *, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1.0 - val_fraction)))
    return idx[:cut], idx[cut:]


@dataclass
class TrainHistory:
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, list]:
        return {
            "epoch": list(self.epoch),
            "train_loss": list(self.train_loss),
            "train_acc": list(self.train_acc),
            "val_loss": list(self.val_loss),
            "val_acc": list(self.val_acc),
        }


def train_session(
    project_dir: Path | str,
    session_ids: List[str],
    *,
    spec: Optional[DatasetSpec] = None,
    cfg: Optional[TrainConfig] = None,
    on_epoch: Optional[callable] = None,
) -> ModelMeta:
    """Build dataset, train the model, evaluate, save to the registry.

    Returns the saved ``ModelMeta``.
    """
    spec = spec or DatasetSpec()
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    ds_all = build_dataset(project_dir, session_ids, spec=spec)
    if ds_all.X.shape[0] < 8:
        raise ValueError(
            f"too few labeled windows ({ds_all.X.shape[0]}). Collect more before training."
        )
    device = _pick_device(cfg.device)
    model_cfg = ModelConfig(
        n_features=ds_all.extractor.n_features,
        sequence_length=spec.sequence_length,
    )
    model = TemporalCSIModel(model_cfg).to(device)

    train_idx, val_idx = _split_indices(ds_all.X.shape[0], cfg.val_fraction, seed=cfg.seed)
    train_ds = _SeqDataset(ds_all.X[train_idx], ds_all.y[train_idx], ds_all.motion[train_idx])
    val_ds = _SeqDataset(ds_all.X[val_idx], ds_all.y[val_idx], ds_all.motion[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = TrainHistory()
    best_val_acc = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tloss, tcorr, tn = 0.0, 0, 0
        for X, y, m in train_loader:
            X, y, m = X.to(device), y.to(device), m.to(device)
            optim.zero_grad()
            logits, motion = model(X)
            loss_occ = F.cross_entropy(logits, y)
            loss_motion = F.mse_loss(motion, m)
            loss = loss_occ + cfg.motion_weight * loss_motion
            loss.backward()
            optim.step()
            tloss += loss.item() * X.shape[0]
            tcorr += (logits.argmax(dim=1) == y).sum().item()
            tn += X.shape[0]
        model.eval()
        vloss, vcorr, vn = 0.0, 0, 0
        with torch.no_grad():
            for X, y, m in val_loader:
                X, y, m = X.to(device), y.to(device), m.to(device)
                logits, motion = model(X)
                loss_occ = F.cross_entropy(logits, y)
                loss_motion = F.mse_loss(motion, m)
                loss = loss_occ + cfg.motion_weight * loss_motion
                vloss += loss.item() * X.shape[0]
                vcorr += (logits.argmax(dim=1) == y).sum().item()
                vn += X.shape[0]
        train_acc = tcorr / max(tn, 1)
        val_acc = vcorr / max(vn, 1)
        history.epoch.append(epoch)
        history.train_loss.append(tloss / max(tn, 1))
        history.train_acc.append(train_acc)
        history.val_loss.append(vloss / max(vn, 1))
        history.val_acc.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        if on_epoch:
            on_epoch(epoch, history)

    # Load the best weights for final eval.
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_result = evaluate(model, ds_all, device=device)

    meta = save_model(
        project_dir=project_dir,
        model=model,
        extractor=ds_all.extractor,
        cfg=model_cfg,
        train_cfg=cfg,
        session_ids=session_ids,
        history=history.to_dict(),
        eval_result=eval_result,
        created_at=now_iso(),
        n_params=count_parameters(model),
    )
    return meta
