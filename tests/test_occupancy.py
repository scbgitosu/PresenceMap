import unittest
from types import SimpleNamespace

from shared.occupancy import (
    OccupancyStateMachine,
    extract_feature_row,
    label_row,
    model_is_ready,
    score_occupancy,
    train_profile_model,
)


def _window(window_id, rssi, snr, *, status="ok", phase="label_vacant"):
    return SimpleNamespace(
        window_id=window_id,
        phase=phase,
        visible_bssid_count=8,
        status=status,
        error_message="" if status != "failed" else "scan failed",
        summary={
            "session_id": "home_occupancy",
            "timestamp_start": f"2026-05-17T00:00:{window_id[-2:]}",
            "timestamp_end": f"2026-05-17T00:00:{window_id[-2:]}",
            "sample_count": 5,
            "rssi_avg_dbm": rssi,
            "rssi_min_dbm": rssi - 2,
            "rssi_max_dbm": rssi + 2,
            "rssi_std_db": 1.2,
            "noise_avg_dbm": -92,
            "snr_avg_db": snr,
            "snr_min_db": snr - 1,
            "tx_bitrate_avg_mbps": 144.4,
            "rx_bitrate_avg_mbps": 144.4,
            "tx_mcs_mode": 7,
            "rx_mcs_mode": 7,
            "neighbor_count_same_channel": 1,
            "neighbor_count_adjacent": 2,
            "neighbor_rssi_sum_dbm": -78,
            "channel_utilization_proxy": 9,
            "missing_sample_count": 0,
        },
    )


class OccupancyTests(unittest.TestCase):
    def test_extract_feature_row_adds_rolling_deltas(self):
        first = extract_feature_row(_window("w01", -55, 35), [], motion_score=0.2)
        second = extract_feature_row(_window("w02", -58, 31), [first], motion_score=0.4)

        self.assertEqual(first["motion_score"], 0.2)
        self.assertEqual(second["rssi_delta_prev_db"], -3)
        self.assertEqual(second["snr_delta_prev_db"], -4)
        self.assertEqual(second["rssi_recent_mean_dbm"], -55)
        self.assertEqual(second["recent_motion_score"], 0.2)

    def test_label_row_maps_training_labels(self):
        feature = extract_feature_row(_window("w01", -55, 35), [], motion_score=None)

        occupied = label_row("occupied_moving", "block-1", feature)
        validation = label_row("validation", "block-2", feature)

        self.assertEqual(occupied["occupancy_label"], "occupied")
        self.assertEqual(occupied["is_training"], "1")
        self.assertEqual(validation["occupancy_label"], "unknown")
        self.assertEqual(validation["is_training"], "0")

    def test_train_and_score_conservative_model_with_hysteresis(self):
        history = []
        feature_rows = []
        label_rows = []
        examples = [
            ("w01", -55, 35, "vacant"),
            ("w02", -56, 34, "vacant"),
            ("w03", -66, 24, "occupied_still"),
            ("w04", -67, 23, "occupied_moving"),
        ]
        for window_id, rssi, snr, label in examples:
            feature = extract_feature_row(_window(window_id, rssi, snr), history, motion_score=0.1)
            feature_rows.append(feature)
            label_rows.append(label_row(label, "block", feature))
            history.append(feature)

        model = train_profile_model(feature_rows, label_rows, session_id="home_occupancy")
        model["thresholds"]["unknown_threshold"] = 0.2
        state_machine = OccupancyStateMachine(required_windows=2)

        self.assertTrue(model_is_ready(model))

        occupied_feature = extract_feature_row(_window("w05", -67, 23), history, motion_score=0.2)
        first = score_occupancy(occupied_feature, model, state_machine)
        second = score_occupancy(occupied_feature, model, state_machine)

        self.assertEqual(first.raw_state, "occupied")
        self.assertEqual(first.stable_state, "unknown")
        self.assertEqual(second.stable_state, "occupied")

    def test_failed_scan_scores_unknown(self):
        feature = extract_feature_row(_window("w01", -55, 35, status="failed"), [], motion_score=None)
        model = train_profile_model([], [], session_id="home_occupancy")

        decision = score_occupancy(feature, model, OccupancyStateMachine())

        self.assertEqual(decision.raw_state, "unknown")
        self.assertEqual(decision.reason, "scan_failed")


if __name__ == "__main__":
    unittest.main()
