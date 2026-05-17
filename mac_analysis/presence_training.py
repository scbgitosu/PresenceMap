"""Train and evaluate PresenceMap occupancy models from synced HP sessions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

_mac_analysis = Path(__file__).resolve().parent
_repo_root = _mac_analysis.parent
sys.path.insert(0, str(_repo_root))

from shared.occupancy import (
    EVAL_ERROR_COLUMNS,
    evaluate_profile_model,
    model_is_ready,
    train_profile_model,
)


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def session_dir(project: Path, session: str) -> Path:
    return Path(project) / "presence_sessions" / session


def train_and_evaluate(
    *,
    project: Path,
    session: str,
    model_path: Optional[Path] = None,
    eval_path: Optional[Path] = None,
    errors_path: Optional[Path] = None,
) -> dict:
    root = session_dir(project, session)
    features_path = root / "presence_features.csv"
    labels_path = root / "presence_labels.csv"
    model_path = model_path or root / "presence_model.json"
    eval_path = eval_path or root / "presence_eval.json"
    errors_path = errors_path or root / "presence_eval_errors.csv"

    feature_rows = _read_csv_rows(features_path)
    label_rows = _read_csv_rows(labels_path)
    if not feature_rows:
        raise RuntimeError(f"No feature rows found at {features_path}")
    if not label_rows:
        raise RuntimeError(f"No label rows found at {labels_path}")

    model = train_profile_model(feature_rows, label_rows, session_id=session)
    evaluation = evaluate_profile_model(feature_rows, label_rows, model, session_id=session)
    evaluation["model_ready"] = model_is_ready(model)
    evaluation["features_path"] = str(features_path)
    evaluation["labels_path"] = str(labels_path)
    evaluation["model_path"] = str(model_path)
    evaluation["errors_path"] = str(errors_path)

    _write_json(model_path, model)
    _write_json(eval_path, evaluation)
    _write_csv(errors_path, EVAL_ERROR_COLUMNS, evaluation.get("errors", []))
    return evaluation


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Train and evaluate a PresenceMap occupancy session")
    parser.add_argument("--project", type=Path, required=True, help="Project directory under survey_projects/")
    parser.add_argument("--session", required=True, help="Presence session name")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional output path for presence_model.json")
    parser.add_argument("--eval-path", type=Path, default=None, help="Optional output path for presence_eval.json")
    parser.add_argument("--errors-path", type=Path, default=None, help="Optional output path for error windows CSV")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    evaluation = train_and_evaluate(
        project=args.project,
        session=args.session,
        model_path=args.model_path,
        eval_path=args.eval_path,
        errors_path=args.errors_path,
    )
    print(f"model_ready={evaluation['model_ready']}")
    print(f"accuracy={evaluation['accuracy']}")
    print(f"known_rate={evaluation['known_rate']}")
    print(f"errors={evaluation['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
