# Architecture

PresenceMap v2 is split across two hosts:

- **HP laptop (Linux)** runs the **sensor agent** ([hp_agent/](../hp_agent/)).
  It owns the AR9271 USB Wi-Fi adapter (monitor mode, CSI capture via the
  Atheros CSI Tool) and the webcam. It does no ML.
- **Mac (Apple Silicon)** runs the **training + inference + dashboard**
  ([mac_app/](../mac_app/)). YOLO labels frames, a small GRU sequence model
  is trained on MPS, the same model runs live.

The two halves talk only over **ZeroMQ + msgpack** on the LAN (or via
Tailscale). No HTTP between hosts; FastAPI only serves Mac-local consumers.

## Process diagram

```
HP                                              Mac
─────                                           ─────
presence-agent run                              presence-mac dashboard
  hp_agent.runners.window_loop                    Streamlit
    iw scan -> rssi_iw -> RFSummary                 mac_app.recorder.SessionRecorder
    CSI frames buffered (csi_buffer)                  mac_app.transport.zmq_subscriber.Subscriber
    aggregate_window_features                        mac_app.yolo.detector.YoloDetector (MPS)
    publish WindowMsg/HealthMsg ───►  PUB 5555 ─► SUB 5555  (window+health)
  hp_agent.runners.frame_loop                       mac_app.yolo.labeler.label_windows
    cv2 -> JPEG ──────────────────►   PUB 5556 ─► SUB 5556  (frames, RCVHWM=1)
                                                    mac_app.train.dataset.SessionParquetWriter
                                                       rf_windows.parquet
                                                       labels.parquet
                                                       yolo_frames.parquet
                                                       thumbnails/*.jpg
                                                       health.parquet

                                                presence-mac train
                                                  build_dataset -> StandardScaler -> sliding T=8
                                                  TemporalCSIModel (Linear -> Conv1d -> GRU -> heads)
                                                  evaluate -> save_model
                                                     data/models/<id>/{weights.pt, meta.json, eval.json}

                                                presence-mac api  (separate uvicorn process)
                                                  FastAPI -> LiveBuffer (SQLite WAL)

                                                Live mode in dashboard
                                                  mac_app.inference.runtime.InferenceLoop
                                                    Subscriber (same as recorder)
                                                    load_model + extractor
                                                    StateSmoother (3-of-5)
                                                    write_prediction(PredictionRow)
                                                                 │
                                                                 ▼
                                                  data/runtime/live_buffer.sqlite3
                                                                 │  (WAL, multi-reader)
                                                                 ▼
                                                  FastAPI server reads here
```

## Wire schemas

All on-wire messages are msgpack dicts (or 4-part bytes for frames).
Source of truth: [hp_agent/transport/messages.py](../hp_agent/transport/messages.py).

| Schema | Topic | Direction | Cadence |
|---|---|---|---|
| `presence.window.v2`  | `window`  | HP → Mac | every `window_seconds` (default 2 s) |
| `presence.health.v2`  | `health`  | HP → Mac | every window + on events |
| `presence.frame.v2`   | `frame`   | HP → Mac | 5 Hz (multipart JPEG) |
| `presence.control.v2` | (REQ/REP) | Mac → HP | on demand (Stage 4+) |

WindowMsg payload:

```
{
  schema: "presence.window.v2",
  agent_id, session_id, window_id,
  ts_start, ts_end, phase, interface, backend,
  rssi: { target_rssi_avg_dbm, target_rssi_std_db, snr_db, noise_dbm,
          visible_bssid_count, neighbor_rssi_sum_dbm,
          channel_utilization_proxy, target_seen },
  csi:  { features:[~174 floats], frames, loss_ratio,
          amp_matrix_f16: <bytes, optional, training-only> }
}
```

## On-disk schema

Source of truth: [shared/schemas.py](../shared/schemas.py) +
[mac_app/train/dataset.py](../mac_app/train/dataset.py).

```
data/survey_projects/<project>/
  project_config.json, floorplan.png, rooms.json, router_positions.json
  presence_sessions/                  # v1, read-only via legacy_v1
  sessions/<session_id>/
    session.json                      # session-level metadata
    rf_windows.parquet                # one row per WindowMsg
    labels.parquet                    # one row per labeled window
    yolo_frames.parquet               # per-frame YOLO metadata
    thumbnails/<frame_id>.jpg
    thumbnails/index.parquet
    health.parquet                    # per-window HealthMsgs
    agent.log.jsonl, mac.log.jsonl

data/models/<model_id>/
  weights.pt, meta.json, eval.json

data/runtime/
  live_buffer.sqlite3                 # shared by dashboard + api
```

## Reliability fixes baked in

v1 had a handful of *silent* failure modes. v2 makes each one observable:

| v1 failure | v2 surfacing |
|---|---|
| `iw link` silently returned "Not connected" -> SNR always NULL | `hp_agent/capture/rssi_iw.py` parses per-BSS `signal:` from scan; reads noise from ath9k debugfs or `iw survey dump`; logs a WARNING when either is unavailable. |
| Target SSID hit-rate <10% with no warning | `hp_agent/health/heartbeat.py` emits `target_ssid_low_visibility` when the 60-window hit rate dips below 50%. |
| OpenCV missing mid-session, never restored | `hp_agent/capture/webcam_stream.py` resolves an index at startup (preflight refuses to run otherwise) and retries every 30 s on mid-session loss; emits `camera_lost`. Never fakes labels. |
| `presence_webcam_calibration.json` missing -> labels written without thresholds | v2 writes `session.json` *before* any windows land; phase gating enforces it. |
| Motion-based webcam labels degenerate (75% vacant) | v2 uses YOLO person detection (`mac_app/yolo/`); manual phase overrides win over YOLO when needed. |

## Live mode flow

1. User picks a model in the dashboard, presses *Start live*.
2. `InferenceLoop` subscribes to the same ZMQ stream the recorder uses.
3. Each `WindowMsg` -> feature extraction (same code as training) -> scaler
   -> rolling sequence of T=8 windows -> `TemporalCSIModel.forward()` on MPS
   -> softmax for occupancy + sigmoid for motion intensity.
4. `StateSmoother` applies 3-of-5 hysteresis to produce a stable state.
5. The row is written to `data/runtime/live_buffer.sqlite3` (WAL mode).
6. The FastAPI process reads from that same SQLite, so dashboard + API share
   state without an in-process bus.

## Code map

| Concern | Files |
|---|---|
| Sensor: RF | [hp_agent/capture/rssi_iw.py](../hp_agent/capture/rssi_iw.py), [hp_agent/capture/csi_atheros.py](../hp_agent/capture/csi_atheros.py), [hp_agent/capture/csi_features.py](../hp_agent/capture/csi_features.py) |
| Sensor: webcam | [hp_agent/capture/webcam_stream.py](../hp_agent/capture/webcam_stream.py) |
| Sensor: orchestration | [hp_agent/runners/window_loop.py](../hp_agent/runners/window_loop.py), [hp_agent/runners/frame_loop.py](../hp_agent/runners/frame_loop.py), [hp_agent/runners/csi_buffer.py](../hp_agent/runners/csi_buffer.py) |
| Transport | [hp_agent/transport/](../hp_agent/transport/), [mac_app/transport/](../mac_app/transport/) |
| Mac: recording | [mac_app/recorder.py](../mac_app/recorder.py), [mac_app/train/dataset.py](../mac_app/train/dataset.py) |
| Mac: YOLO | [mac_app/yolo/](../mac_app/yolo/) |
| Mac: training | [mac_app/train/{features,model,train,eval,registry}.py](../mac_app/train/) |
| Mac: inference | [mac_app/inference/](../mac_app/inference/) |
| Mac: API | [mac_app/api/](../mac_app/api/) |
| Mac: dashboard | [mac_app/dashboard/](../mac_app/dashboard/) |
| Shared | [shared/](../shared/) |
