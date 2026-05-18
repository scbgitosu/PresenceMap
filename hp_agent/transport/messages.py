"""Msgpack-encoded wire messages between HP agent and Mac.

Schemas are simple dicts (not dataclasses) — easier to evolve, easier to
encode/decode without a schema-registry layer. Each carries ``schema`` and
``schema_version`` so downstream code can route them safely.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import msgpack

from shared.versioning import SCHEMA_VERSION

# Topic strings (bytes on the wire — ZMQ filters bytes prefixes)
TOPIC_WINDOW = b"window"
TOPIC_HEALTH = b"health"
TOPIC_FRAME = b"frame"

WINDOW_SCHEMA = "presence.window.v2"
HEALTH_SCHEMA = "presence.health.v2"
FRAME_SCHEMA = "presence.frame.v2"
CONTROL_SCHEMA = "presence.control.v2"


def encode(payload: Dict[str, Any]) -> bytes:
    """Pack a Python dict to msgpack bytes."""
    return msgpack.packb(payload, use_bin_type=True)


def decode(buf: bytes) -> Dict[str, Any]:
    """Unpack msgpack bytes to a Python dict."""
    return msgpack.unpackb(buf, raw=False)


def make_window_msg(
    *,
    agent_id: str,
    session_id: str,
    window_id: str,
    ts_start: str,
    ts_end: str,
    phase: str,
    rssi: Dict[str, Any],
    csi: Optional[Dict[str, Any]] = None,
    interface: str = "",
    backend: str = "",
) -> Dict[str, Any]:
    return {
        "schema": WINDOW_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id,
        "session_id": session_id,
        "window_id": window_id,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "phase": phase,
        "rssi": rssi,
        "csi": csi or {"features": [], "frames": 0, "loss_ratio": 1.0},
        "interface": interface,
        "backend": backend,
    }


def make_health_msg(
    *,
    agent_id: str,
    session_id: str,
    ts: str,
    code: str,
    detail: str = "",
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema": HEALTH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id,
        "session_id": session_id,
        "ts": ts,
        "code": code,
        "detail": detail,
        "metrics": metrics or {},
    }


def make_frame_parts(
    *,
    agent_id: str,
    ts_us: int,
    jpeg: bytes,
) -> Iterable[bytes]:
    """Multipart frame message: [TOPIC, agent_id_bytes, ts_us_bytes, jpeg_bytes].

    Kept as a tuple of bytes so callers can ``socket.send_multipart(list(...))``.
    """
    return (
        TOPIC_FRAME,
        agent_id.encode("utf-8"),
        ts_us.to_bytes(8, "big", signed=False),
        jpeg,
    )


def parse_frame_parts(parts) -> Dict[str, Any]:
    """Inverse of make_frame_parts."""
    _topic, agent_id, ts_us_bytes, jpeg = parts
    return {
        "schema": FRAME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "agent_id": agent_id.decode("utf-8"),
        "ts_us": int.from_bytes(ts_us_bytes, "big", signed=False),
        "jpeg": bytes(jpeg),
    }


def make_control_msg(cmd: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema": CONTROL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cmd": cmd,
        "args": args or {},
    }
