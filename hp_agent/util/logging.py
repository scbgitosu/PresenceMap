"""Structured JSON logging for the HP agent.

Writes one JSON object per line to ``<session>/agent.log.jsonl`` and also
emits a human-readable record to stderr (colorized when stderr is a TTY).
Level is controlled by the ``PRESENCE_LOG_LEVEL`` environment variable
(default ``INFO``).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_TTY_COLORS = {
    "DEBUG": "\x1b[2m",
    "INFO": "\x1b[0m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
}
_RESET = "\x1b[0m"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        lvl = record.levelname
        head = f"{ts} {lvl:<7} {record.name}"
        if self._use_color:
            color = _TTY_COLORS.get(lvl, "")
            head = f"{color}{head}{_RESET}"
        return f"{head} {record.getMessage()}"


def get_logger(name: str, *, session_log: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger. Idempotent per ``(name, session_log)`` pair."""
    logger = logging.getLogger(name)
    # Idempotent: don't double-attach handlers.
    marker = f"_presence_configured::{session_log}"
    if getattr(logger, marker, False):  # type: ignore[arg-type]
        return logger
    level_str = os.environ.get("PRESENCE_LOG_LEVEL", "INFO").upper()
    logger.setLevel(_LEVELS.get(level_str, logging.INFO))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ConsoleFormatter(use_color=sys.stderr.isatty()))
    logger.addHandler(console)

    if session_log is not None:
        session_log = Path(session_log)
        session_log.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(session_log, mode="a")
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)

    setattr(logger, marker, True)  # type: ignore[arg-type]
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, msg: str, **fields: object) -> None:
    """Emit a log record carrying structured fields for the JSONL sink."""
    lvl = _LEVELS.get(level.upper(), logging.INFO)
    logger.log(lvl, msg, extra={"extra_fields": fields})
