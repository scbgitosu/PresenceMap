"""TemporalCSIModel: small CSI sequence classifier.

Architecture (see plan, Section 4.2):

    Input:                  (B, T, F)
    -> Linear(F -> H)       per-step projection
    -> Conv1d(H, H, k=3)    short temporal smoothing
    -> ReLU
    -> GRU(H, H)            recurrence over the sequence
    -> [last hidden state]  (B, H)
    -> Linear(H -> 2)       occupancy logits
    -> Linear(H -> 1) + sigmoid   motion intensity (auxiliary regression head)

Total params for F=196, H=64, T=8: ~60k. Trains in seconds-to-minutes on MPS.
The same module is used for training and inference -- no ONNX/CoreML export.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_features: int
    sequence_length: int = 8
    hidden: int = 64
    conv_kernel: int = 3
    dropout: float = 0.1


class TemporalCSIModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.n_features, cfg.hidden)
        pad = cfg.conv_kernel // 2
        self.conv = nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=cfg.conv_kernel, padding=pad)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_occ = nn.Linear(cfg.hidden, 2)
        self.head_motion = nn.Linear(cfg.hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``(B, T, F)``. Returns ``(occ_logits[B,2], motion[B])``."""
        h = self.proj(x)                        # (B, T, H)
        h_t = h.transpose(1, 2)                 # (B, H, T)
        h_t = F.relu(self.conv(h_t))
        h = h_t.transpose(1, 2)                 # (B, T, H)
        out, _ = self.gru(h)                    # (B, T, H)
        last = self.dropout(out[:, -1, :])      # (B, H)
        logits = self.head_occ(last)            # (B, 2)
        motion = torch.sigmoid(self.head_motion(last)).squeeze(-1)  # (B,)
        return logits, motion


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
