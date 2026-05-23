# Roadmap

## v3 (current) — ESP32 on Mac

- ADR-018 UDP ingest with per-node diagnostics
- Streamlit UI: ESP32 Nodes, Train, Models
- Phase-labeled sessions (`labeled_vacant` / `labeled_occupied`)
- `TemporalCSIModel` live inference via fused multi-node features
- Stubs: per-room floorplan occupancy, automation scenes

## Next

1. **Bedroom presence stable** — seq-gap alarms, ingest + inference in one supervised process option
2. **Distributed nodes** — one ESP32 per room; multi-head or per-room models + `rooms.json` heatmap
3. **Automation** — scene recorder, MQTT / Home Assistant webhooks
4. **Research** — vitals / sleep metrics (honest confidence bands), optional camera-assisted training

## Retired (removed in v3 cleanup)

- HeatMap survey / heatmap placement tools
- HP Linux `presence-agent` + ZMQ transport
- AR9271 Atheros CSI tool + `iw` RSSI path
- YOLO webcam labeling pipeline
- Legacy v1 session viewer
