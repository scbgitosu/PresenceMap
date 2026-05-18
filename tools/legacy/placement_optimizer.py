"""
Fit a path-loss model from router-position trials and suggest new AP coordinates.

Usage:
    python3 mac_analysis/placement_optimizer.py \
        --project survey_projects/apartment_test \
        --sessions trial_pos1,trial_pos2,trial_pos3 \
        --output-dir output/placement
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from shared.path_loss import (
    build_training_rows,
    distance_ft,
    fit_path_loss,
    infer_room_id,
    leave_one_session_out_rmse,
    predict_rssi,
)
from shared.survey_metrics import (
    DEFAULT_GOOD_DBM,
    build_room_mask,
    discover_sessions,
    load_measurements,
    load_rooms,
    load_routers,
)
from shared.utils import project_paths

from mac_analysis.floorplan_viz import annotate_placement_heatmap_ax, plot_placement_overview


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _router_map(routers: list) -> dict:
    return {r["router_position_id"]: r for r in routers}


def _candidate_grid(width: int, height: int, rooms: list, step_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(0, width, step_px)
    ys = np.arange(0, height, step_px)
    grid_x, grid_y = np.meshgrid(xs, ys)
    if rooms:
        mask = build_room_mask(rooms, grid_x, grid_y)
    else:
        mask = np.ones(grid_x.shape, dtype=bool)
    return grid_x, grid_y, mask


def _nearest_labeled_distance(x: float, y: float, routers: list) -> float:
    if not routers:
        return float("inf")
    return min(((x - r["x_px"]) ** 2 + (y - r["y_px"]) ** 2) ** 0.5 for r in routers)


def _score_candidate(
    model,
    ap_x: float,
    ap_y: float,
    ap_height_ft: float,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    receiver_mask: np.ndarray,
    rooms: list,
    scale_pixels_per_foot: float,
    good_dbm: float,
) -> dict:
    ap_room = infer_room_id(ap_x, ap_y, rooms)
    values = []
    for rx, ry, in_room in zip(receiver_x.ravel(), receiver_y.ravel(), receiver_mask.ravel()):
        if not in_room:
            continue
        rx_room = infer_room_id(float(rx), float(ry), rooms)
        dist = distance_ft(ap_x, ap_y, ap_height_ft, float(rx), float(ry), 4.0, scale_pixels_per_foot)
        values.append(predict_rssi(model, dist, cross_room=bool(ap_room and rx_room and ap_room != rx_room)))
    arr = np.array(values)
    if arr.size == 0:
        return {"score": 0.0, "pct_area_good": 0.0, "p10_dbm": None, "mean_dbm": None}
    pct_good = 100.0 * float(np.sum(arr >= good_dbm)) / len(arr)
    p10 = float(np.percentile(arr, 10))
    mean = float(np.mean(arr))
    score = 0.65 * pct_good + 0.35 * max(0.0, min(100.0, (p10 + 85.0) * 4.0))
    return {"score": score, "pct_area_good": pct_good, "p10_dbm": p10, "mean_dbm": mean}


def _predict_grid(
    model,
    ap_x: float,
    ap_y: float,
    ap_height_ft: float,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    receiver_mask: np.ndarray,
    rooms: list,
    scale_pixels_per_foot: float,
) -> np.ndarray:
    ap_room = infer_room_id(ap_x, ap_y, rooms)
    grid = np.full(receiver_x.shape, np.nan)
    for idx, (rx, ry, in_room) in enumerate(zip(receiver_x.ravel(), receiver_y.ravel(), receiver_mask.ravel())):
        if not in_room:
            continue
        rx_room = infer_room_id(float(rx), float(ry), rooms)
        dist = distance_ft(ap_x, ap_y, ap_height_ft, float(rx), float(ry), 4.0, scale_pixels_per_foot)
        grid.ravel()[idx] = predict_rssi(model, dist, cross_room=bool(ap_room and rx_room and ap_room != rx_room))
    return grid


def _plot_predicted_coverage(
    grid: np.ndarray,
    image_path: Path,
    output_path: Path,
    title: str,
    *,
    routers: list,
    candidates: list,
):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    cmap = mcolors.ListedColormap(["#cc0000", "#ff4500", "#ff8c00", "#ffd700", "#90ee00", "#00c800"])
    norm = mcolors.BoundaryNorm([-100, -75, -70, -67, -60, -50, 0], cmap.N)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(np.array(img), origin="upper", zorder=0)
    im = ax.imshow(
        grid,
        origin="upper",
        extent=[0, w, h, 0],
        cmap=cmap,
        norm=norm,
        alpha=0.55,
        zorder=5,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Predicted RSSI (dBm)", fontsize=10)
    annotate_placement_heatmap_ax(ax, routers, candidates)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def optimize_placement(
    project_dir: Path,
    session_ids: list[str],
    output_dir: Path,
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
    candidate_step_px: int = 80,
    receiver_step_px: int = 80,
    top_k: int = 5,
    ap_height_ft: float = 4.0,
):
    paths = project_paths(project_dir)
    metadata = _load_metadata(paths["floorplan_metadata"])
    scale = metadata.get("scale_pixels_per_foot")
    if not scale:
        raise ValueError(
            "Missing scale_pixels_per_foot. Re-run floorplan_import.py and measure a known wall length."
        )
    scale = float(scale)

    rooms = load_rooms(paths["rooms_json"])
    routers = load_routers(paths["router_positions_json"])
    routers_by_id = _router_map(routers)
    measurements_by_session = {
        sid: load_measurements(paths["survey_sessions_dir"] / sid / "measurements_summary.csv", sid)
        for sid in session_ids
    }
    training_rows = build_training_rows(measurements_by_session, routers_by_id, rooms, scale)
    model = fit_path_loss(training_rows)
    loso_rmse = leave_one_session_out_rmse(training_rows)

    img = Image.open(paths["floorplan_png"]).convert("RGB")
    width, height = img.size
    cand_x, cand_y, cand_mask = _candidate_grid(width, height, rooms, candidate_step_px)
    rx_x, rx_y, rx_mask = _candidate_grid(width, height, rooms, receiver_step_px)

    candidates = []
    for x, y, in_room in zip(cand_x.ravel(), cand_y.ravel(), cand_mask.ravel()):
        if not in_room:
            continue
        if _nearest_labeled_distance(float(x), float(y), routers) < 50:
            continue
        metrics = _score_candidate(
            model,
            float(x),
            float(y),
            ap_height_ft,
            rx_x,
            rx_y,
            rx_mask,
            rooms,
            scale,
            good_dbm,
        )
        candidates.append({
            "x_px": float(x),
            "y_px": float(y),
            "height_ft": ap_height_ft,
            **metrics,
        })

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]
    for i, candidate in enumerate(ranked, start=1):
        candidate["rank"] = i

    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "p0_dbm": model.p0_dbm,
        "path_loss_exponent_n": model.exponent_n,
        "cross_room_penalty_db": model.cross_room_penalty_db,
        "rmse_db": model.rmse_db,
        "leave_one_session_out_rmse_db": loso_rmse,
        "sample_count": model.sample_count,
        "scale_pixels_per_foot": scale,
    }
    with open(output_dir / "model_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {output_dir / 'model_params.json'}")

    recommendation = {
        "good_threshold_dbm": good_dbm,
        "candidate_step_px": candidate_step_px,
        "receiver_step_px": receiver_step_px,
        "top_candidates": ranked,
    }
    with open(output_dir / "placement_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Saved: {output_dir / 'placement_recommendation.json'}")

    overview_path = output_dir / "placement_overview.png"
    plot_placement_overview(
        paths["floorplan_png"],
        routers,
        ranked,
        overview_path,
        title="Trial AP locations (orange) and suggested placements (green #1)",
    )

    predicted_path = None
    if ranked:
        best = ranked[0]
        grid = _predict_grid(
            model,
            best["x_px"],
            best["y_px"],
            best["height_ft"],
            rx_x,
            rx_y,
            rx_mask,
            rooms,
            scale,
        )
        predicted_path = output_dir / "predicted_coverage_rank1.png"
        _plot_predicted_coverage(
            grid,
            paths["floorplan_png"],
            predicted_path,
            f"Predicted coverage if AP is at green #1 ({best['x_px']:.0f}, {best['y_px']:.0f})",
            routers=routers,
            candidates=ranked,
        )

    print("\nPath-loss model")
    print(f"  P0                  : {model.p0_dbm:.1f} dBm")
    print(f"  Exponent n          : {model.exponent_n:.2f}")
    print(f"  Cross-room penalty  : {model.cross_room_penalty_db:.1f} dB")
    print(f"  Fit RMSE            : {model.rmse_db:.1f} dB")
    if loso_rmse is not None:
        print(f"  Leave-one-session RMSE: {loso_rmse:.1f} dB")
    if ranked:
        best = ranked[0]
        print(
            f"\nBest suggested coordinate: ({best['x_px']:.0f}, {best['y_px']:.0f}) "
            f"score {best['score']:.1f}, {best['pct_area_good']:.0f}% area good"
        )
    return {
        "model_params": params,
        "recommendation": recommendation,
        "artifacts": {
            "model_params_json": output_dir / "model_params.json",
            "placement_recommendation_json": output_dir / "placement_recommendation.json",
            "placement_overview_png": overview_path,
            "predicted_coverage_png": predicted_path,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Suggest router/AP placement from path-loss trials")
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--sessions", default="", help="Comma-separated training session IDs")
    parser.add_argument("--output-dir", default="output/placement")
    parser.add_argument("--good-threshold", type=float, default=DEFAULT_GOOD_DBM)
    parser.add_argument("--candidate-step-px", type=int, default=80)
    parser.add_argument("--receiver-step-px", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ap-height-ft", type=float, default=4.0)
    args = parser.parse_args()

    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    if args.sessions.strip():
        session_ids = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        session_ids = discover_sessions(paths["survey_sessions_dir"])
    if not session_ids:
        print("No survey sessions found.")
        sys.exit(1)

    try:
        optimize_placement(
            project_dir,
            session_ids,
            Path(args.output_dir),
            good_dbm=args.good_threshold,
            candidate_step_px=args.candidate_step_px,
            receiver_step_px=args.receiver_step_px,
            top_k=args.top_k,
            ap_height_ft=args.ap_height_ft,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
