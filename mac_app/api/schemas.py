"""Pydantic schemas for the FastAPI live state service."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class StateResponse(BaseModel):
    occupancy: str
    raw: str
    confidence: float
    motion_intensity: float
    model_id: str
    session_id: str
    window_id: str
    ts_start: str
    ts_end: str
    as_of: str


class HistoryPoint(BaseModel):
    window_id: str
    session_id: str
    ts_start: str
    ts_end: str
    raw: str
    stable: str
    confidence: float
    motion_intensity: float
    model_id: str
    received_at: str


class HistoryResponse(BaseModel):
    points: List[HistoryPoint]
    since: Optional[str] = None
    until: Optional[str] = None
    limit: int


class HealthResponse(BaseModel):
    agent_status: str
    last_window_age_s: float
    signal_loss: bool
    last_health_code: Optional[str] = None
    last_health_detail: Optional[str] = None
    last_health_ts: Optional[str] = None
    active_model_id: Optional[str] = None


class ModelEntry(BaseModel):
    model_id: str
    created_at: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    session_ids: List[str] = []


class ModelsResponse(BaseModel):
    models: List[ModelEntry]
    active_model_id: Optional[str] = None


class SetActiveModelRequest(BaseModel):
    model_id: str


class SetActiveModelResponse(BaseModel):
    model_id: str
    set_at: str
