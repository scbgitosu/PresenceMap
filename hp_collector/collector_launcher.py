"""
Button-style launcher for the HP field collector.

Usage:
    python3 hp_collector/collector_launcher.py
    python3 hp_collector/collector_launcher.py --project survey_projects/apartment_test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "survey_projects/apartment_test"


class CollectorLauncher(QMainWindow):
    def __init__(self, project: str):
        super().__init__()
        self.setWindowTitle("Wi-Fi Survey Collector Launcher")
        self.process: QProcess | None = None
        self._build_ui(project)

    def _build_ui(self, project: str):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        layout.addWidget(QLabel("Project path:"))
        row = QHBoxLayout()
        self.project_edit = QLineEdit(project)
        self.project_edit.editingFinished.connect(self._update_presence_output_hint)
        row.addWidget(self.project_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_project)
        row.addWidget(browse_btn)
        load_cfg_btn = QPushButton("Load Presence Config")
        load_cfg_btn.clicked.connect(self._load_presence_config)
        row.addWidget(load_cfg_btn)
        layout.addLayout(row)

        button_row = QHBoxLayout()
        preflight_btn = QPushButton("Run Preflight")
        preflight_btn.clicked.connect(self._run_preflight)
        button_row.addWidget(preflight_btn)

        open_btn = QPushButton("Open Project Folder")
        open_btn.clicked.connect(self._open_project_folder)
        button_row.addWidget(open_btn)
        layout.addLayout(button_row)

        presence_grp = QGroupBox("Bedroom Presence Training Rig")
        presence_layout = QVBoxLayout(presence_grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Presence session name:"))
        self.presence_session_edit = QLineEdit("bedroom_v1")
        self.presence_session_edit.setToolTip("Folder name under <project>/presence_sessions/. Do not include survey_sessions/.")
        self.presence_session_edit.editingFinished.connect(self._update_presence_output_hint)
        row.addWidget(self.presence_session_edit)
        row.addWidget(QLabel("Interface:"))
        self.presence_interface_edit = QLineEdit("wlan1")
        row.addWidget(self.presence_interface_edit)
        row.addWidget(QLabel("Room:"))
        self.presence_location_edit = QLineEdit("bedroom")
        row.addWidget(self.presence_location_edit)
        presence_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Baseline seconds:"))
        self.baseline_seconds_edit = QLineEdit("300")
        row.addWidget(self.baseline_seconds_edit)
        row.addWidget(QLabel("Training seconds:"))
        self.training_seconds_edit = QLineEdit("1200")
        row.addWidget(self.training_seconds_edit)
        row.addWidget(QLabel("Validation seconds:"))
        self.validation_seconds_edit = QLineEdit("600")
        row.addWidget(self.validation_seconds_edit)
        presence_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label source:"))
        self.label_source_edit = QLineEdit("webcam_derived")
        row.addWidget(self.label_source_edit)
        row.addWidget(QLabel("Source detail:"))
        self.label_source_detail_edit = QLineEdit("derived occupancy only; no continuous video retained")
        row.addWidget(self.label_source_detail_edit)
        presence_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Collector placement:"))
        self.collector_placement_edit = QLineEdit("HP + AR9271 fixed in bedroom")
        row.addWidget(self.collector_placement_edit)
        row.addWidget(QLabel("Router placement:"))
        self.router_placement_edit = QLineEdit("router fixed in bedroom")
        row.addWidget(self.router_placement_edit)
        presence_layout.addLayout(row)

        workflow_row = QHBoxLayout()
        smoke_btn = QPushButton("Smoke Test")
        smoke_btn.clicked.connect(self._presence_smoke_test)
        workflow_row.addWidget(smoke_btn)

        baseline_btn = QPushButton("1 Calibrate Vacant")
        baseline_btn.clicked.connect(self._presence_calibrate)
        workflow_row.addWidget(baseline_btn)

        vacant_btn = QPushButton("2 Train Vacant")
        vacant_btn.clicked.connect(lambda: self._presence_label_block("vacant", self._training_seconds()))
        workflow_row.addWidget(vacant_btn)

        still_btn = QPushButton("3 Train Occupied Still")
        still_btn.clicked.connect(lambda: self._presence_label_block("occupied_still", self._training_seconds()))
        workflow_row.addWidget(still_btn)

        moving_btn = QPushButton("4 Train Occupied Moving")
        moving_btn.clicked.connect(lambda: self._presence_label_block("occupied_moving", self._validation_seconds()))
        workflow_row.addWidget(moving_btn)
        presence_layout.addLayout(workflow_row)

        validation_row = QHBoxLayout()
        val_vacant_btn = QPushButton("5 Validate Vacant")
        val_vacant_btn.clicked.connect(lambda: self._presence_label_block("validation_vacant", self._validation_seconds()))
        validation_row.addWidget(val_vacant_btn)

        val_occupied_btn = QPushButton("6 Validate Occupied")
        val_occupied_btn.clicked.connect(lambda: self._presence_label_block("validation_occupied", self._validation_seconds()))
        validation_row.addWidget(val_occupied_btn)

        monitor_btn = QPushButton("Run Live Monitor")
        monitor_btn.clicked.connect(self._presence_monitor)
        validation_row.addWidget(monitor_btn)

        stop_btn = QPushButton("Stop Running Step")
        stop_btn.clicked.connect(self._stop_process)
        validation_row.addWidget(stop_btn)
        presence_layout.addLayout(validation_row)

        self.presence_output_label = QLabel("")
        presence_layout.addWidget(self.presence_output_label)
        presence_layout.addWidget(QLabel("Mac analysis after syncing: python3 mac_analysis/presence_training.py --project <project> --session <presence-session>"))
        layout.addWidget(presence_grp)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(360)
        layout.addWidget(self.output)

        self.resize(820, 520)
        self._load_presence_config(quiet=True)
        self._update_presence_output_hint()

    def _project(self) -> str:
        return self.project_edit.text().strip() or DEFAULT_PROJECT

    def _presence_session(self, *, quiet: bool = False) -> str:
        raw = self.presence_session_edit.text().strip() or "bedroom_v1"
        normalized = Path(raw).name
        if normalized != raw:
            self.presence_session_edit.setText(normalized)
            if not quiet:
                self._append(f"[presence] Using session name '{normalized}' from '{raw}'")
        return normalized

    def _presence_output_dir(self) -> Path:
        return Path(self._project()) / "presence_sessions" / self._presence_session(quiet=True)

    def _update_presence_output_hint(self):
        if hasattr(self, "presence_output_label"):
            self.presence_output_label.setText(f"Output folder: {self._presence_output_dir()}")

    def _project_path(self) -> Path:
        project_path = Path(self._project())
        return project_path if project_path.is_absolute() else REPO_ROOT / project_path

    def _load_presence_config(self, quiet: bool = False):
        project_path = self._project_path()
        config_path = project_path / "project_config.json"
        if not config_path.exists():
            if not quiet:
                self._append(f"[presence] No project_config.json found at {config_path}")
            self._update_presence_output_hint()
            return
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as e:
            self._append(f"[presence] Could not read {config_path}: {e}")
            return

        presence = config.get("presence", {})
        if config.get("default_interface"):
            self.presence_interface_edit.setText(config["default_interface"])
        room_label = presence.get("first_room_label") or "bedroom"
        self.presence_location_edit.setText(room_label)
        if presence.get("default_session"):
            self.presence_session_edit.setText(presence["default_session"])
        else:
            self.presence_session_edit.setText(f"{room_label}_v1")
        if presence.get("label_source"):
            self.label_source_edit.setText(presence["label_source"])
        if presence.get("label_source_detail"):
            self.label_source_detail_edit.setText(presence["label_source_detail"])
        elif presence.get("video_retention"):
            self.label_source_detail_edit.setText(f"{presence['video_retention']}; no continuous video retained")
        if presence.get("collector_placement"):
            self.collector_placement_edit.setText(presence["collector_placement"])
        if presence.get("router_placement"):
            self.router_placement_edit.setText(presence["router_placement"])

        if not quiet:
            room = self._presence_location()
            session = self._presence_session(quiet=True)
            iface = self._presence_interface()
            self._append(f"[presence] Loaded config: room={room} session={session} interface={iface}")
        self._update_presence_output_hint()

    def _presence_interface(self) -> str:
        return self.presence_interface_edit.text().strip() or "wlan1"

    def _presence_location(self) -> str:
        return self.presence_location_edit.text().strip() or "bedroom"

    def _seconds(self, edit: QLineEdit, default: str) -> str:
        value = edit.text().strip() or default
        try:
            seconds = max(1, int(float(value)))
        except ValueError:
            seconds = int(default)
        edit.setText(str(seconds))
        return str(seconds)

    def _baseline_seconds(self) -> str:
        return self._seconds(self.baseline_seconds_edit, "300")

    def _training_seconds(self) -> str:
        return self._seconds(self.training_seconds_edit, "1200")

    def _validation_seconds(self) -> str:
        return self._seconds(self.validation_seconds_edit, "600")

    def _append(self, text: str):
        self.output.append(text.rstrip())

    def _run_process(self, program: str, arguments: list[str], label: str):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, label, "A step is already running. Stop it before starting another.")
            return

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._collector_finished)
        self._append(f"$ {program} {' '.join(arguments)}")
        self.process.start()

    def _stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self._append("[stop requested]")
            self.process.terminate()
        else:
            self._append("[no running step]")

    def _browse_project(self):
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose survey project",
            str((REPO_ROOT / self._project()).resolve()),
        )
        if chosen:
            self.project_edit.setText(chosen)
            self._load_presence_config()

    def _run_preflight(self):
        project = self._project()
        cmd = [
            sys.executable,
            "hp_collector/preflight.py",
            "--project",
            project,
            "--interface",
            self._presence_interface(),
        ]
        self._append(f"$ {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as e:
            self._append(f"ERROR: {e}")
            return
        if result.stdout:
            self._append(result.stdout)
        if result.stderr:
            self._append(result.stderr)
        if result.returncode == 0:
            QMessageBox.information(self, "Preflight", "Wi-Fi preflight passed.")
        else:
            QMessageBox.warning(self, "Preflight", "Wi-Fi preflight failed. See output.")

    def _presence_base_args(self) -> list[str]:
        return [
            "--project",
            self._project(),
            "--session",
            self._presence_session(),
            "--interface",
            self._presence_interface(),
            "--location-label",
            self._presence_location(),
            "--collector-placement",
            self.collector_placement_edit.text().strip(),
            "--router-placement",
            self.router_placement_edit.text().strip(),
            "--webcam-placement",
            f"{self._presence_location()} overview for derived labels only",
            "--label-source",
            self.label_source_edit.text().strip() or "webcam_derived",
            "--label-source-detail",
            self.label_source_detail_edit.text().strip() or "derived occupancy only; no continuous video retained",
        ]

    def _run_presence(self, extra_args: list[str], label: str):
        script = REPO_ROOT / "scripts" / "run_presence_tripwire.sh"
        self._run_process(str(script), self._presence_base_args() + extra_args, label)

    def _presence_smoke_test(self):
        self._run_presence(
            [
                "--monitor",
                "--max-windows",
                "3",
                "--backend",
                "auto",
                "--samples-per-window",
                "3",
                "--delay",
                "1.0",
                "--window-seconds",
                "8",
            ],
            "Presence smoke test",
        )

    def _presence_calibrate(self):
        self._run_presence(["--calibrate", "--baseline-seconds", self._baseline_seconds()], "Presence calibration")

    def _presence_label_block(self, label: str, seconds: str):
        self._run_presence(["--label-block", label, "--block-seconds", seconds], f"Presence {label}")

    def _presence_monitor(self):
        self._run_presence(["--monitor", "--occupancy-monitor"], "Presence live monitor")

    def _read_stdout(self):
        if self.process:
            self._append(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _read_stderr(self):
        if self.process:
            self._append(bytes(self.process.readAllStandardError()).decode(errors="replace"))

    def _collector_finished(self, code: int, _status):
        self._append(f"[collector exited with code {code}]")

    def _open_project_folder(self):
        project_path = Path(self._project())
        if not project_path.is_absolute():
            project_path = REPO_ROOT / project_path
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(project_path)])
            else:
                subprocess.Popen(["xdg-open", str(project_path)])
        except Exception as e:
            self._append(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Launch the HP Wi-Fi collector")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Survey project directory")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = CollectorLauncher(args.project)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
