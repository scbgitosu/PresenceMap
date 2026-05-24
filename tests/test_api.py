"""End-to-end tests for the FastAPI live state service."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from mac_app.inference.live_buffer import LiveBuffer, PredictionRow


def _setup_buffer(tmp_path: Path) -> Path:
    """Prepare an isolated LiveBuffer SQLite + populate it with two predictions."""
    db = tmp_path / "live.sqlite3"
    buf = LiveBuffer(db)
    buf.write_prediction(
        PredictionRow(
            window_id="w001",
            session_id="s",
            ts_start="2026-05-18T00:00:01.000Z",
            ts_end="2026-05-18T00:00:03.000Z",
            raw="vacant",
            stable="vacant",
            confidence=0.91,
            motion_intensity=0.05,
            model_id="m_test",
            received_at="2026-05-18T00:00:03.000Z",
        )
    )
    buf.write_prediction(
        PredictionRow(
            window_id="w002",
            session_id="s",
            ts_start="2026-05-18T00:00:03.000Z",
            ts_end="2026-05-18T00:00:05.000Z",
            raw="occupied",
            stable="occupied",
            confidence=0.88,
            motion_intensity=0.6,
            model_id="m_test",
            received_at="2026-05-18T00:00:05.000Z",
        )
    )
    buf.set_meta("active_model_id", "m_test")
    buf.write_health({
        "ts": "2026-05-18T00:00:04.000Z",
        "agent_id": "hp-1",
        "session_id": "s",
        "code": "ok",
        "detail": "heartbeat",
        "metrics": {"loss_ratio": 0.0},
    })
    buf.close()
    return db


def _make_app(db_path: Path, project_dir: Path):
    os.environ["PRESENCE_SQLITE_PATH"] = str(db_path)
    os.environ["PRESENCE_PROJECT_DIR"] = str(project_dir)
    # Force a fresh import so create_app() picks up the env vars.
    import importlib

    import mac_app.api.server as server_module

    server_module = importlib.reload(server_module)
    return server_module.create_app()


def test_state_endpoint(tmp_path: Path) -> None:
    db = _setup_buffer(tmp_path)
    app = _make_app(db, tmp_path)
    with TestClient(app) as c:
        r = c.get("/state")
        assert r.status_code == 200
        body = r.json()
        assert body["occupancy"] == "occupied"
        assert body["model_id"] == "m_test"
        assert body["window_id"] == "w002"


def test_history_endpoint(tmp_path: Path) -> None:
    db = _setup_buffer(tmp_path)
    app = _make_app(db, tmp_path)
    with TestClient(app) as c:
        r = c.get("/history", params={"limit": 10})
        assert r.status_code == 200
        body = r.json()
        assert len(body["points"]) == 2
        assert body["points"][0]["window_id"] == "w001"


def test_history_since_filter(tmp_path: Path) -> None:
    db = _setup_buffer(tmp_path)
    app = _make_app(db, tmp_path)
    with TestClient(app) as c:
        r = c.get(
            "/history",
            params={"since": "2026-05-18T00:00:04.000Z", "limit": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["points"]) == 1
        assert body["points"][0]["window_id"] == "w002"


def test_health_endpoint(tmp_path: Path) -> None:
    db = _setup_buffer(tmp_path)
    app = _make_app(db, tmp_path)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["active_model_id"] == "m_test"
        assert body["last_health_code"] == "ok"
        assert body["last_health_detail"] == "heartbeat"


def test_set_active_model(tmp_path: Path) -> None:
    db = _setup_buffer(tmp_path)
    app = _make_app(db, tmp_path)
    with TestClient(app) as c:
        r = c.post("/models/active", json={"model_id": "new_model"})
        assert r.status_code == 200
        assert r.json()["model_id"] == "new_model"
        r2 = c.get("/health")
        assert r2.status_code == 200
        assert r2.json()["active_model_id"] == "new_model"
