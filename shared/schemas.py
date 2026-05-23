"""Session / window schema notes for ESP32 parquet columns."""

# rf_windows.parquet (ESP32)
#   window_id, session_id, ts_start, ts_end, phase, node_id, source_ip,
#   csi_features (list[float]), csi_frames, csi_loss_ratio, backend

# labels.parquet
#   window_id, occupancy (vacant|occupied), label_source (manual|esp32_phase)
