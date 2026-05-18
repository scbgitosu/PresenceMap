"""Schema and model format versions.

Bump when the wire/file format changes in a way that older readers can't handle.
``SCHEMA_VERSION`` covers WindowMsg / parquet columns; ``MODEL_VERSION`` covers
the trained-model JSON payload.
"""
from __future__ import annotations

SCHEMA_VERSION: int = 2
MODEL_VERSION: int = 1
