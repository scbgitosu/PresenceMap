"""v2 column lists for parquet artifacts.

The authoritative source is ``hp_agent.transport.messages`` (WindowMsg etc.) and
the dataset writers in ``mac_app.train.dataset``; this module just collects the
column-name constants so writers and readers agree.

Stage 1 stub: only the names are defined. Stages 2–4 fill in types as the
writers come online.
"""
from __future__ import annotations

# rf_windows.parquet
RF_WINDOW_COLUMNS = [
    "window_id",
    "session_id",
    "agent_id",
    "ts_start",
    "ts_end",
    "phase",
    "csi_features",       # list<float>
    "csi_frames",         # int
    "csi_loss_ratio",     # float
    "csi_amp_matrix_f16", # binary, nullable
    "target_rssi_avg_dbm",
    "target_rssi_std_db",
    "snr_db",
    "noise_dbm",
    "visible_bssid_count",
    "neighbor_rssi_sum_dbm",
    "channel_utilization_proxy",
    "target_seen",        # bool
    "interface",
    "backend",
    "schema_version",
]

# labels.parquet
LABEL_COLUMNS = [
    "window_id",
    "session_id",
    "ts_start",
    "ts_end",
    "occupancy",          # vacant | occupied | unknown
    "person_count",
    "label_confidence",
    "label_source",       # yolo | manual | missing
    "yolo_frame_refs",    # list<str>
    "yolo_model_id",
    "note",
    "schema_version",
]

# yolo_frames.parquet
YOLO_FRAME_COLUMNS = [
    "frame_id",
    "ts",
    "session_id",
    "agent_id",
    "jpeg_path",          # relative; nullable after purge
    "person_count",
    "max_conf",
    "bboxes",             # list<{cls, conf, x1, y1, x2, y2}>
    "yolo_model_id",
    "schema_version",
]

# thumbnails/index.parquet
THUMBNAIL_INDEX_COLUMNS = [
    "frame_id",
    "ts",
    "path",
    "person_count",
    "kept_reason",        # rate | state_change | manual | purged
    "schema_version",
]

# health.parquet
HEALTH_COLUMNS = [
    "ts",
    "session_id",
    "agent_id",
    "code",               # ok | target_ssid_low_visibility | camera_lost | csi_loss_high | scan_failed | disk_full | window_stalled
    "detail",
    "metrics_json",
    "schema_version",
]

OCCUPANCY_LABELS = ("vacant", "occupied", "unknown")
LABEL_SOURCES = ("yolo", "manual", "missing")
