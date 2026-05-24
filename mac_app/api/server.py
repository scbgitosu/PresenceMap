"""FastAPI live state service.

Reads from the same SQLite-backed ``LiveBuffer`` the inference loop writes to,
so the dashboard and the API can run in separate processes and stay in sync.

Configure via env vars:
    PRESENCE_SQLITE_PATH   path to live_buffer.sqlite3 (default: data/runtime/...)
    PRESENCE_PROJECT_DIR   project root used for /models registry lookup
    PRESENCE_API_BIND      uvicorn host:port (default: 127.0.0.1:8765)

Endpoints:
    GET  /state             latest smoothed prediction
    GET  /history           paginated window list
    GET  /health            agent + inference health
    GET  /models            list registered models
    POST /models/active     set active model id (localhost only)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from mac_app.api.schemas import (
    HealthResponse,
    HistoryPoint,
    HistoryResponse,
    ModelEntry,
    ModelsResponse,
    SetActiveModelRequest,
    SetActiveModelResponse,
    StateResponse,
)
from mac_app.inference.live_buffer import LiveBuffer, default_sqlite_path
from mac_app.train.registry import list_models
from shared.utils import now_iso


def _sqlite_path() -> Path:
    p = os.environ.get("PRESENCE_SQLITE_PATH")
    return Path(p) if p else default_sqlite_path()


def _project_dir() -> Path:
    p = os.environ.get("PRESENCE_PROJECT_DIR")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[2] / "data" / "survey_projects" / "apartment_test"


def create_app() -> FastAPI:
    app = FastAPI(title="PresenceMap Live API", version="2.0")
    buffer = LiveBuffer(_sqlite_path())
    project_dir = _project_dir()

    @app.get("/state", response_model=StateResponse)
    def get_state():
        latest = buffer.latest_prediction()
        if latest is None:
            raise HTTPException(status_code=503, detail="no predictions yet")
        return StateResponse(
            occupancy=latest.stable,
            raw=latest.raw,
            confidence=latest.confidence,
            motion_intensity=latest.motion_intensity,
            model_id=latest.model_id,
            session_id=latest.session_id,
            window_id=latest.window_id,
            ts_start=latest.ts_start,
            ts_end=latest.ts_end,
            as_of=latest.received_at,
        )

    @app.get("/history", response_model=HistoryResponse)
    def get_history(
        since: Optional[str] = Query(None),
        until: Optional[str] = Query(None),
        limit: int = Query(1000, ge=1, le=10_000),
    ):
        rows = buffer.predictions(since=since, until=until, limit=limit)
        return HistoryResponse(
            points=[
                HistoryPoint(
                    window_id=r.window_id,
                    session_id=r.session_id,
                    ts_start=r.ts_start,
                    ts_end=r.ts_end,
                    raw=r.raw,
                    stable=r.stable,
                    confidence=r.confidence,
                    motion_intensity=r.motion_intensity,
                    model_id=r.model_id,
                    received_at=r.received_at,
                )
                for r in rows
            ],
            since=since,
            until=until,
            limit=limit,
        )

    @app.get("/health", response_model=HealthResponse)
    def get_health():
        recent = buffer.recent_health(limit=1)
        last_health = recent[0] if recent else None
        latest = buffer.latest_prediction()
        last_age = float("inf")
        if latest is not None:
            from datetime import datetime, timezone

            try:
                received = datetime.fromisoformat(latest.received_at.replace("Z", "+00:00"))
                last_age = (datetime.now(timezone.utc) - received).total_seconds()
            except Exception:
                last_age = float("inf")
        signal_lost = last_age > 6.0
        return HealthResponse(
            agent_status="ok" if not signal_lost else "signal_lost",
            last_window_age_s=last_age if last_age != float("inf") else -1.0,
            signal_loss=signal_lost,
            last_health_code=last_health["code"] if last_health else None,
            last_health_detail=last_health["detail"] if last_health else None,
            last_health_ts=last_health["ts"] if last_health else None,
            active_model_id=buffer.get_meta("active_model_id"),
        )

    @app.get("/models", response_model=ModelsResponse)
    def get_models():
        models = list_models(project_dir)
        return ModelsResponse(
            models=[
                ModelEntry(
                    model_id=m.model_id,
                    created_at=m.created_at,
                    accuracy=m.eval_summary.get("accuracy"),
                    precision=m.eval_summary.get("precision"),
                    recall=m.eval_summary.get("recall"),
                    f1=m.eval_summary.get("f1"),
                    session_ids=m.session_ids,
                )
                for m in models
            ],
            active_model_id=buffer.get_meta("active_model_id"),
        )

    @app.post("/models/active", response_model=SetActiveModelResponse)
    def set_active_model(req: SetActiveModelRequest, request: Request):
        # Localhost guard. Default API bind is 127.0.0.1 so non-local callers can't
        # reach us in practice; this is a belt-and-braces check for the case where
        # the operator has flipped PRESENCE_API_BIND to 0.0.0.0. `testclient` is
        # the host FastAPI's TestClient reports.
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(status_code=403, detail="localhost-only endpoint")
        buffer.set_meta("active_model_id", req.model_id)
        return SetActiveModelResponse(model_id=req.model_id, set_at=now_iso())

    return app


app = create_app()


def main(*, host: Optional[str] = None, port: Optional[int] = None) -> int:
    import uvicorn

    bind = os.environ.get("PRESENCE_API_BIND", "127.0.0.1:8765")
    if ":" in bind and host is None and port is None:
        host_s, port_s = bind.rsplit(":", 1)
        host = host_s or "127.0.0.1"
        port = int(port_s)
    host = host or "127.0.0.1"
    port = port or 8765
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
