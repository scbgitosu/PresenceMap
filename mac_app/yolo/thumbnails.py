"""Rolling 320x240 thumbnail writer + post-session purger.

Writes thumbnails at most ``target_fps`` (default 1 Hz) to
``<session_dir>/thumbnails/<ts>.jpg``. On session stop, ``purge_session``
keeps one thumbnail per minute plus one per occupancy state-change boundary
(other thumbnails get deleted, but their rows stay in
``thumbnails/index.parquet`` with ``kept_reason="purged"``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2  # type: ignore[import-not-found]
import numpy as np


@dataclass
class ThumbnailConfig:
    target_fps: float = 1.0
    width: int = 320
    height: int = 240
    jpeg_quality: int = 70


class ThumbnailWriter:
    def __init__(self, session_dir: Path, cfg: Optional[ThumbnailConfig] = None) -> None:
        self.dir = Path(session_dir) / "thumbnails"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ThumbnailConfig()
        self._last_ts_us: Optional[int] = None

    def maybe_write(
        self,
        frame_bgr: np.ndarray,
        ts_us: int,
        *,
        frame_id: str,
    ) -> Optional[Tuple[Path, int, int]]:
        """If enough time has passed since the last thumbnail, save one.

        Returns ``(path, width, height)`` when a thumbnail was written, else
        ``None``.
        """
        min_gap_us = int(1_000_000 / max(self.cfg.target_fps, 0.1))
        if self._last_ts_us is not None and (ts_us - self._last_ts_us) < min_gap_us:
            return None
        resized = cv2.resize(frame_bgr, (self.cfg.width, self.cfg.height))
        path = self.dir / f"{frame_id}.jpg"
        ok = cv2.imwrite(
            str(path),
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality],
        )
        if not ok:
            return None
        self._last_ts_us = ts_us
        return path, self.cfg.width, self.cfg.height


def purge_session(
    session_dir: Path,
    *,
    keep_per_minute: int = 1,
    keep_state_changes: bool = True,
    thumbnails_index: Optional[List[dict]] = None,
) -> List[dict]:
    """Delete thumbnails that aren't worth keeping.

    Returns an updated index list with ``kept_reason="purged"`` for each
    deletion. If ``thumbnails_index`` is None the function just walks the
    directory and produces a fresh index.
    """
    thumbs_dir = Path(session_dir) / "thumbnails"
    if not thumbs_dir.exists():
        return thumbnails_index or []
    # Build a (ts, path) list either from the index or from the directory.
    if thumbnails_index is None:
        files = sorted(thumbs_dir.glob("*.jpg"))
        thumbnails_index = [
            {
                "frame_id": p.stem,
                "ts": p.stat().st_mtime_ns // 1000,
                "path": str(p.relative_to(session_dir)),
                "person_count": 0,
                "kept_reason": "rate",
            }
            for p in files
        ]
    keep_set = set()
    by_minute: dict = {}
    last_person_count: Optional[int] = None
    for row in thumbnails_index:
        ts_us = int(row.get("ts") or 0)
        minute = ts_us // (60 * 1_000_000)
        bucket = by_minute.setdefault(minute, [])
        bucket.append(row)
    for minute, rows in by_minute.items():
        rows.sort(key=lambda r: r["ts"])
        for r in rows[:keep_per_minute]:
            keep_set.add(r["frame_id"])
            r["kept_reason"] = "rate"
    if keep_state_changes:
        last_pc: Optional[int] = None
        for row in sorted(thumbnails_index, key=lambda r: r["ts"]):
            pc = int(row.get("person_count") or 0)
            if last_pc is not None and pc != last_pc:
                keep_set.add(row["frame_id"])
                row["kept_reason"] = "state_change"
            last_pc = pc
    # Apply.
    for row in thumbnails_index:
        if row["frame_id"] in keep_set:
            continue
        p = Path(session_dir) / row["path"]
        try:
            if p.exists():
                p.unlink()
            row["kept_reason"] = "purged"
        except OSError:
            pass
    return thumbnails_index
