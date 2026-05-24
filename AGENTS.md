# AGENTS.md — PresenceMap

ESP32 CSI room-presence on Mac. RuView firmware on ESP32-S3 nodes; PresenceMap owns ingest, UI, training, and live inference.

## Repository

- Path: `/Users/scottbrough/Projects/PresenceMap`
- [RuView/](RuView/): upstream reference (firmware, ADR-018). Do not treat as PresenceMap application code.
- HeatMap upstream remote (if present): `heatmap-upstream` — do not push PresenceMap work there.

## Architecture (v3)

- **Mac only** for application code: `mac_app/`, `shared/`, `docs/`, `tests/`
- **No** `hp_agent/`, AR9271, ZMQ HP transport, or YOLO labeling
- Ingest: `presence-mac esp32-ingest` → `data/runtime/esp32_state.json`
- Training: `sessions/*/rf_windows.parquet` with manual phase labels

## Product rules

- Do not claim clinical vitals, person ID, or pose “out of the box”
- Diagnostics before occupancy; show LIVE vs missing ingest honestly
- Only one process may bind UDP port 5005

## GitNexus

Use GitNexus impact analysis before editing symbols in `mac_app/` or `shared/`. Run `gitnexus_detect_changes()` before commits.

See embedded GitNexus block below for MCP tools.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **PresenceMap**. Use GitNexus MCP tools for impact analysis and navigation.

<!-- gitnexus:end -->
