import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from mac_analysis.presence_training import train_and_evaluate
from shared.occupancy import (
    OccupancyStateMachine,
    evaluate_profile_model,
    extract_feature_row,
    label_row,
    model_is_ready,
    score_occupancy,
    train_profile_model,
)
from shared.webcam_ground_truth import (
    WebcamOccupancyState,
    recommend_motion_threshold,
    summarize_calibration_samples,
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
        derived = label_row(
            "validation_occupied",
            "block-3",
            feature,
            label_source="webcam_derived",
            source_detail="frame_count_only",
            label_confidence="0.9",
        )

        self.assertEqual(occupied["occupancy_label"], "occupied")
        self.assertEqual(occupied["is_training"], "1")
        self.assertEqual(validation["occupancy_label"], "unknown")
        self.assertEqual(validation["is_training"], "0")
        self.assertEqual(derived["occupancy_label"], "occupied")
        self.assertEqual(derived["is_training"], "0")
        self.assertEqual(derived["label_source"], "webcam_derived")

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

    def test_evaluate_profile_model_reports_errors(self):
        history = []
        feature_rows = []
        label_rows = []
        training = [
            ("w01", -55, 35, "vacant"),
            ("w02", -56, 34, "vacant"),
            ("w03", -66, 24, "occupied_still"),
            ("w04", -67, 23, "occupied_moving"),
        ]
        for window_id, rssi, snr, label in training:
            feature = extract_feature_row(_window(window_id, rssi, snr), history, motion_score=0.1)
            feature_rows.append(feature)
            label_rows.append(label_row(label, "train", feature))
            history.append(feature)

        validation_feature = extract_feature_row(_window("w05", -66, 24), history, motion_score=0.2)
        feature_rows.append(validation_feature)
        label_rows.append(
            label_row(
                "validation_occupied",
                "validation",
                validation_feature,
                label_source="webcam_derived",
            )
        )

        mismatch_feature = extract_feature_row(_window("w06", -55, 35), history, motion_score=0.0)
        feature_rows.append(mismatch_feature)
        label_rows.append(label_row("validation_occupied", "validation", mismatch_feature))

        model = train_profile_model(feature_rows, label_rows, session_id="home_occupancy")
        model["thresholds"]["unknown_threshold"] = 0.2

        evaluation = evaluate_profile_model(feature_rows, label_rows, model, session_id="home_occupancy")

        self.assertEqual(evaluation["total_labeled_windows"], 6)
        self.assertGreaterEqual(evaluation["correct_windows"], 5)
        self.assertEqual(evaluation["error_count"], 1)
        self.assertEqual(evaluation["errors"][0]["window_id"], "w06")

    def test_train_and_evaluate_writes_artifacts(self):
        history = []
        feature_rows = []
        label_rows = []
        examples = [
            ("w01", -55, 35, "vacant"),
            ("w02", -56, 34, "vacant"),
            ("w03", -66, 24, "occupied"),
            ("w04", -67, 23, "occupied"),
        ]
        for window_id, rssi, snr, label in examples:
            feature = extract_feature_row(_window(window_id, rssi, snr), history, motion_score=0.1)
            feature_rows.append(feature)
            label_rows.append(label_row(label, "block", feature))
            history.append(feature)

        with TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            session_dir = project / "presence_sessions" / "home_occupancy"
            session_dir.mkdir(parents=True)
            features_path = session_dir / "presence_features.csv"
            labels_path = session_dir / "presence_labels.csv"

            import csv

            with open(features_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
                writer.writeheader()
                writer.writerows(feature_rows)
            with open(labels_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(label_rows[0].keys()))
                writer.writeheader()
                writer.writerows(label_rows)

            evaluation = train_and_evaluate(project=project, session="home_occupancy")

            self.assertTrue((session_dir / "presence_model.json").exists())
            self.assertTrue((session_dir / "presence_eval.json").exists())
            self.assertTrue((session_dir / "presence_eval_errors.csv").exists())
            self.assertTrue(evaluation["model_ready"])

    def test_webcam_state_derives_motion_hold_and_vacancy(self):
        state = WebcamOccupancyState(hold_seconds=10)
        moving = state.derive(
            observed_at=100,
            motion_ratio=0.04,
            frame_count=5,
            brightness_avg=80,
            min_frames=3,
            motion_threshold=0.015,
            low_light_threshold=25,
            source_detail="derived occupancy only; no continuous video retained",
        )
        still = state.derive(
            observed_at=105,
            motion_ratio=0.0,
            frame_count=5,
            brightness_avg=80,
            min_frames=3,
            motion_threshold=0.015,
            low_light_threshold=25,
            source_detail="derived occupancy only; no continuous video retained",
        )
        vacant = state.derive(
            observed_at=120,
            motion_ratio=0.0,
            frame_count=5,
            brightness_avg=80,
            min_frames=3,
            motion_threshold=0.015,
            low_light_threshold=25,
            source_detail="derived occupancy only; no continuous video retained",
        )

        self.assertEqual(moving.label, "occupied_moving")
        self.assertEqual(still.label, "occupied_still")
        self.assertEqual(vacant.label, "vacant")

    def test_webcam_unknown_labels_do_not_train_model(self):
        history = []
        feature_rows = []
        label_rows = []
        for window_id, rssi, snr, label in [
            ("w01", -55, 35, "vacant"),
            ("w02", -56, 34, "vacant"),
            ("w03", -66, 24, "occupied"),
            ("w04", -67, 23, "occupied"),
            ("w05", -60, 30, "unknown"),
        ]:
            feature = extract_feature_row(_window(window_id, rssi, snr), history, motion_score=0.1)
            feature_rows.append(feature)
            label_rows.append(label_row(label, "webcam", feature, label_source="webcam_derived"))
            history.append(feature)

        model = train_profile_model(feature_rows, label_rows, session_id="home_occupancy")

        self.assertEqual(model["profiles"]["vacant"]["window_count"], 2)
        self.assertEqual(model["profiles"]["occupied"]["window_count"], 2)
        self.assertTrue(model_is_ready(model))

    def test_webcam_calibration_summarizes_and_recommends_threshold(self):
        empty = summarize_calibration_samples([
            {"brightness_avg": 70, "motion_ratio": 0.002, "occupancy_label": "vacant"},
            {"brightness_avg": 72, "motion_ratio": 0.004, "occupancy_label": "vacant"},
        ])
        person = summarize_calibration_samples([
            {"brightness_avg": 68, "motion_ratio": 0.03, "occupancy_label": "occupied"},
            {"brightness_avg": 69, "motion_ratio": 0.05, "occupancy_label": "occupied"},
        ])
        threshold = recommend_motion_threshold(empty, person)

        self.assertEqual(empty["sample_count"], 2)
        self.assertEqual(empty["motion_max"], 0.004)
        self.assertEqual(person["motion_mean"], 0.04)
        self.assertEqual(threshold, 0.022)


if __name__ == "__main__":
    unittest.main()
