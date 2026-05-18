# PresenceMap Roadmap

## v1 (HeatMap fork → tripwire prototype)  ✓ archived

The original headless HP collector (`hp_collector/presence_tripwire.py`) and
the PyQt launchers (`collector_launcher.py`, `collector_app.py`) lived here.
A diagnostic review of the `webcam_master` and `empty_master` sessions
showed structural failures (silent `iw link` errors → 100% NULL SNR,
motion-ratio webcam labels degenerating to 75% vacant, scattered camera
dropouts, missing calibration JSONs). The plan to rebuild lives at
[.claude/plans/check-the-current-implementation-sprightly-pixel.md](../.claude/plans/check-the-current-implementation-sprightly-pixel.md).

The v1 code has been archived under [tools/legacy/](../tools/legacy/) and is
not imported anywhere. v1 session data still on disk is rendered read-only
via [shared/legacy_v1/readers.py](../shared/legacy_v1/readers.py) in the
Legacy Viewer dashboard page.

## v2 (current)  ✓ landed

Six incremental commits on the `v2-rehash` branch:

1. **Repo restructure + legacy archive.** New layout: `hp_agent/`,
   `mac_app/`, `tools/legacy/`, `data/`. Survey-only modules archived.
   `pyproject.toml` declares `presence-agent` and `presence-mac` console
   scripts. `requirements-hp.txt` / `requirements-mac.txt` split.
2. **HP agent skeleton + RSSI/iw SNR fix + ZMQ transport.** `presence-agent
   {preflight, webcam-test, run}` works on Ubuntu 22.04. SNR comes from per-
   BSS scan signal + ath9k noise floor (the v1 `iw link` silent-failure
   path is gone). Target-SSID hit-rate alarms surface low visibility.
   `presence-mac transport-tap` proves the wire end-to-end.
3. **CSI capture + per-window features.** AR9271 monitor mode + Atheros CSI
   Tool. Per-window feature vector: 56 amp_mean + 56 amp_std + 56 phase_std
   (with linear detrend) + 5 scalars + loss ratio. Fixture source enables
   Mac-side dev without the hardware.
4. **Streamlit dashboard (Training mode) + YOLO + parquet.** YOLOv8n on
   MPS at 1–2 Hz, frame→window label alignment, rolling thumbnails,
   v2-schema parquet writes (`rf_windows`, `labels`, `yolo_frames`,
   `thumbnails/index`, `health`). Read-only legacy viewer for v1 sessions.
5. **TemporalCSIModel + training + eval + registry.** Linear → Conv1d →
   GRU sequence model with binary occupancy + sigmoid motion-intensity
   heads. `presence-mac train` CLI + Train page in the dashboard. Models
   saved under `data/models/<model_id>/` with `weights.pt`, `meta.json`,
   `eval.json`. Side-by-side comparison in Models page.
6. **Live mode + FastAPI REST API.** `InferenceLoop` streams predictions
   into a SQLite-backed `LiveBuffer`. Dashboard Live page shows current
   state, confidence, motion intensity, state changes. `presence-mac api`
   serves `/state`, `/history`, `/health`, `/models` on `127.0.0.1:8765`.

## Open questions (v3+)

- **Per-zone occupancy** using `rooms.json` polygons. v2 is binary
  (occupied / vacant); the natural next step is one classifier head per
  zone and a calibration UI that walks the user through each room.
- **Entry/exit events.** Sequence model already sees T=8 windows; a third
  head could predict "entering" / "leaving" / "present" if we collect a
  labeled transition dataset.
- **CSI + RSSI fusion at the input layer** vs. the current concatenation.
  A two-tower model that processes amp / phase separately may help.
- **MQTT + Home Assistant.** Plan keeps these out of v2 for scope reasons;
  add behind a feature flag once the binary path is stable.
- **Streaming parquet writes.** v2 buffers in memory and writes on
  `stop()`; for multi-hour sessions we should switch to ParquetWriter in
  append mode (with crash-safety shards).
