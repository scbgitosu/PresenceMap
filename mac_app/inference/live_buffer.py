"""SQLite-backed ring buffer for live inference state.

Lets the inference loop, Streamlit dashboard, and FastAPI server share state
without each running their own process-local buffer. The same DB file
(``data/runtime/live_buffer.sqlite3`` by default) is opened in WAL mode so
multiple readers + one writer is safe.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.project import runtime_dir
from shared.versioning import SCHEMA_VERSION

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    window_id    TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    ts_start     TEXT NOT NULL,
    ts_end       TEXT NOT NULL,
    raw          TEXT NOT NULL,
    stable       TEXT NOT NULL,
    confidence   REAL NOT NULL,
    motion_intensity REAL NOT NULL,
    model_id     TEXT NOT NULL,
    received_at  TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_received_at_idx ON predictions(received_at);

CREATE TABLE IF NOT EXISTS agent_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    code         TEXT NOT NULL,
    detail       TEXT,
    metrics_json TEXT,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_health_ts_idx ON agent_health(ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class PredictionRow:
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

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class LiveBuffer:
    def __init__(self, sqlite_path: Optional[Path] = None) -> None:
        self.path = Path(sqlite_path) if sqlite_path else default_sqlite_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- writers ---------------------------------------------------------- #

    def write_prediction(self, row: PredictionRow) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    window_id, session_id, ts_start, ts_end,
                    raw, stable, confidence, motion_intensity,
                    model_id, received_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.window_id, row.session_id, row.ts_start, row.ts_end,
                    row.raw, row.stable, row.confidence, row.motion_intensity,
                    row.model_id, row.received_at, SCHEMA_VERSION,
                ),
            )
            self._conn.commit()

    def write_health(self, payload: Dict[str, Any]) -> None:
        import json

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_health (ts, agent_id, session_id, code, detail, metrics_json, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("ts", ""),
                    payload.get("agent_id", ""),
                    payload.get("session_id", ""),
                    payload.get("code", "ok"),
                    payload.get("detail", ""),
                    json.dumps(payload.get("metrics") or {}),
                    SCHEMA_VERSION,
                ),
            )
            self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ---- readers ---------------------------------------------------------- #

    def latest_prediction(self) -> Optional[PredictionRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT window_id, session_id, ts_start, ts_end, raw, stable, "
                "confidence, motion_intensity, model_id, received_at "
                "FROM predictions ORDER BY received_at DESC LIMIT 1"
            ).fetchone()
        if r is None:
            return None
        return PredictionRow(*r)

    def predictions(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 1000,
    ) -> List[PredictionRow]:
        sql = (
            "SELECT window_id, session_id, ts_start, ts_end, raw, stable, "
            "confidence, motion_intensity, model_id, received_at FROM predictions"
        )
        params: list = []
        clauses: List[str] = []
        if since is not None:
            clauses.append("received_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("received_at <= ?")
            params.append(until)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY received_at ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [PredictionRow(*r) for r in rows]

    def recent_health(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, agent_id, session_id, code, detail, metrics_json "
                "FROM agent_health ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"ts": ts, "agent_id": a, "session_id": s, "code": c, "detail": d, "metrics_json": mj}
            for ts, a, s, c, d, mj in rows
        ]

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            r = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def clear_predictions(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM predictions")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def default_sqlite_path() -> Path:
    """``<repo_root>/data/runtime/live_buffer.sqlite3``."""
    here = Path(__file__).resolve().parents[2]
    return runtime_dir(here) / "live_buffer.sqlite3"
