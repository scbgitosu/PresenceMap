"""
Compare Wi-Fi survey sessions (e.g. one walk per router candidate position).

Usage:
    python3 mac_analysis/session_compare.py \\
        --project survey_projects/apartment_test \\
        --output-dir output/comparison

    # Explicit sessions (one full walk per router trial):
    python3 mac_analysis/session_compare.py \\
        --project survey_projects/apartment_test \\
        --sessions trial_pos1,trial_pos2,trial_pos3 \\
        --output-dir output/comparison
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mac_analysis.heatmap_generator import (
    generate_heatmap,
    generate_survey_points,
    generate_weak_zones,
    _load_measurements,
    _load_rooms,
    _load_routers,
)
from mac_analysis.floorplan_viz import draw_trial_routers, plot_router_trial_map
from mac_analysis.walk_matcher import build_walk_pairs, write_walk_pairs
from shared.survey_metrics import (
    DEFAULT_GOOD_DBM,
    DEFAULT_WEAK_DBM,
    SessionMetrics,
    analyze_project,
    discover_sessions,
    interpolate_rssi_grid,
    load_measurements,
)
from shared.utils import project_paths


def _label(sm: SessionMetrics) -> str:
    if sm.router_name and sm.router_name != sm.router_position_id:
        return f"{sm.router_name}\n({sm.session_id})"
    return sm.session_id


def write_metrics_csv(results: list[SessionMetrics], path: Path):
    columns = [
        "rank",
        "session_id",
        "router_position_id",
        "router_name",
        "composite_score",
        "point_count",
        "valid_count",
        "rssi_mean_dbm",
        "rssi_median_dbm",
        "rssi_min_dbm",
        "rssi_p10_dbm",
        "mean_rssi_std_db",
        "snr_mean_db",
        "snr_min_db",
        "tx_bitrate_mean_mbps",
        "rx_bitrate_mean_mbps",
        "neighbor_count_same_channel_mean",
        "neighbor_count_adjacent_mean",
        "channel_utilization_proxy_mean",
        "pct_points_below_good",
        "pct_points_below_weak",
        "pct_points_unstable",
        "pct_area_good",
        "pct_area_weak",
        "rssi_grid_min_dbm",
        "rssi_grid_p10_dbm",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for sm in sorted(results, key=lambda x: x.rank):
            p, g = sm.points, sm.grid
            w.writerow(
                {
                    "rank": sm.rank,
                    "session_id": sm.session_id,
                    "router_position_id": sm.router_position_id,
                    "router_name": sm.router_name,
                    "composite_score": sm.composite_score,
                    "point_count": p.point_count,
                    "valid_count": p.valid_count,
                    "rssi_mean_dbm": _fmt(p.rssi_mean),
                    "rssi_median_dbm": _fmt(p.rssi_median),
                    "rssi_min_dbm": _fmt(p.rssi_min),
                    "rssi_p10_dbm": _fmt(p.rssi_p10),
                    "mean_rssi_std_db": _fmt(p.mean_rssi_std),
                    "snr_mean_db": _fmt(p.snr_mean),
                    "snr_min_db": _fmt(p.snr_min),
                    "tx_bitrate_mean_mbps": _fmt(p.tx_bitrate_mean),
                    "rx_bitrate_mean_mbps": _fmt(p.rx_bitrate_mean),
                    "neighbor_count_same_channel_mean": _fmt(p.neighbor_count_same_channel_mean),
                    "neighbor_count_adjacent_mean": _fmt(p.neighbor_count_adjacent_mean),
                    "channel_utilization_proxy_mean": _fmt(p.channel_utilization_proxy_mean),
                    "pct_points_below_good": _fmt(p.pct_below_good),
                    "pct_points_below_weak": _fmt(p.pct_below_weak),
                    "pct_points_unstable": _fmt(p.pct_unstable),
                    "pct_area_good": _fmt(g.pct_area_good),
                    "pct_area_weak": _fmt(g.pct_area_weak),
                    "rssi_grid_min_dbm": _fmt(g.rssi_grid_min),
                    "rssi_grid_p10_dbm": _fmt(g.rssi_grid_p10),
                }
            )
    print(f"Saved: {path}")


def write_room_csv(results: list[SessionMetrics], path: Path):
    columns = [
        "session_id",
        "router_name",
        "room_id",
        "room_name",
        "point_count",
        "rssi_mean_dbm",
        "rssi_min_dbm",
        "pct_points_below_good",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for sm in results:
            for rm in sm.rooms:
                w.writerow(
                    {
                        "session_id": sm.session_id,
                        "router_name": sm.router_name,
                        "room_id": rm.room_id,
                        "room_name": rm.room_name,
                        "point_count": rm.point_count,
                        "rssi_mean_dbm": _fmt(rm.rssi_mean),
                        "rssi_min_dbm": _fmt(rm.rssi_min),
                        "pct_points_below_good": _fmt(rm.pct_below_good),
                    }
                )
    print(f"Saved: {path}")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def plot_comparison_bars(results: list[SessionMetrics], path: Path):
    labels = [_label(sm) for sm in results]
    x = np.arange(len(results))
    width = 0.22

    area_good = [sm.grid.pct_area_good if sm.grid.valid_cell_count else 100 - sm.points.pct_below_good for sm in results]
    p10 = [sm.points.rssi_p10 or sm.points.rssi_min or -80 for sm in results]
    mean_std = [sm.points.mean_rssi_std or 0 for sm in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(x, area_good, color="#00a050", edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("% area / points ≥ good threshold")
    axes[0].set_title("Coverage (good signal)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 100)

    axes[1].bar(x, p10, color="#4488cc", edgecolor="black", linewidth=0.6)
    axes[1].axhline(-67, color="#888", linestyle="--", linewidth=1, label="−67 dBm")
    axes[1].set_ylabel("dBm")
    axes[1].set_title("Worst-case (10th %ile point RSSI)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].legend(fontsize=8)

    axes[2].bar(x, mean_std, color="#cc8844", edgecolor="black", linewidth=0.6)
    axes[2].set_ylabel("dBm (lower = more stable)")
    axes[2].set_title("Mean RSSI std at click points")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)

    fig.suptitle("Session comparison — router placement trials", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ranking(results: list[SessionMetrics], path: Path):
    ordered = sorted(results, key=lambda s: s.composite_score)
    labels = [_label(sm) for sm in ordered]
    scores = [sm.composite_score for sm in ordered]
    colors = plt.cm.RdYlGn(np.linspace(0.35, 0.9, len(ordered)))

    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.9)))
    y = np.arange(len(ordered))
    ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite score (higher = better placement)")
    ax.set_title("Recommended router position ranking")
    for i, (score, sm) in enumerate(zip(scores, ordered)):
        ax.text(score + 0.5, i, f"#{sm.rank}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_walk_deltas(pairs: list, path: Path):
    if not pairs:
        return
    labels = [f"{p['session_b']} - {p['session_a']}\n{p['waypoint_id']}" for p in pairs]
    values = [
        float(p["delta_rssi_dbm"])
        for p in pairs
        if p.get("delta_rssi_dbm") not in ("", None)
    ]
    if not values:
        return
    plot_labels = [
        labels[i]
        for i, p in enumerate(pairs)
        if p.get("delta_rssi_dbm") not in ("", None)
    ]
    x = np.arange(len(values))
    colors = ["#00a050" if v >= 0 else "#cc3333" for v in values]
    fig, ax = plt.subplots(figsize=(max(10, len(values) * 0.45), 5))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_ylabel("Delta RSSI (dB)")
    ax.set_title("Matched walk point deltas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=70, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coverage_diff(
    results: list[SessionMetrics],
    project_dir: Path,
    rooms: list,
    output_path: Path,
    *,
    good_dbm: float,
    downsample: int = 4,
):
    """Difference map: best session grid minus worst session grid (by composite score)."""
    if len(results) < 2:
        return

    paths = project_paths(project_dir)
    img = Image.open(paths["floorplan_png"]).convert("RGB")
    w, h = img.size

    grids = {}
    for sm in results:
        summary = paths["survey_sessions_dir"] / sm.session_id / "measurements_summary.csv"
        measurements = load_measurements(summary, session_id=sm.session_id)
        grid = interpolate_rssi_grid(measurements, w, h, downsample=downsample, rooms=rooms)
        if grid is not None:
            grids[sm.session_id] = grid

    if len(grids) < 2:
        print("Skipping coverage diff map (need ≥2 sessions with ≥3 points each).")
        return

    best = max(results, key=lambda s: s.composite_score)
    worst = min(results, key=lambda s: s.composite_score)
    if best.session_id not in grids or worst.session_id not in grids:
        return

    diff = grids[best.session_id] - grids[worst.session_id]
    img_arr = np.array(img)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img_arr, origin="upper", zorder=0)
    im = ax.imshow(
        diff,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="RdYlGn",
        vmin=-15,
        vmax=15,
        alpha=0.55,
        zorder=5,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Δ RSSI (dBm): best − worst trial", fontsize=10)
    routers = _load_routers(paths["router_positions_json"])
    draw_trial_routers(ax, routers, highlight_position_id=best.router_position_id)
    ax.set_title(
        f"Coverage gain: {best.router_name or best.session_id} (★ best) vs "
        f"{worst.router_name or worst.session_id}",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_report(results: list[SessionMetrics], *, good_dbm: float, weak_dbm: float):
    if not results:
        print("No sessions to compare.")
        return

    print("\n" + "=" * 72)
    print("ROUTER PLACEMENT COMPARISON")
    print("=" * 72)
    print(
        f"Thresholds: good ≥ {good_dbm:.0f} dBm | weak < {weak_dbm:.0f} dBm | poor < −75 dBm"
    )
    print(
        "\nMetrics: point RSSI (avg/min/std), interpolated area coverage, per-room breakdown."
    )
    print("Composite score weights: grid coverage 40%, worst-case 25%, mean RSSI 20%,")
    print("                           stability 10%, scan reliability 5%.\n")

    for sm in sorted(results, key=lambda s: s.rank):
        p, g = sm.points, sm.grid
        print(f"#{sm.rank}  {sm.router_name or sm.router_position_id}  [{sm.session_id}]")
        print(f"     Composite score     : {sm.composite_score:.1f}")
        if p.valid_count:
            print(
                f"     Points (valid/total): {p.valid_count}/{p.point_count}  "
                f"mean {p.rssi_mean:.1f} dBm  min {p.rssi_min:.1f}  p10 {p.rssi_p10:.1f}"
            )
            if p.mean_rssi_std is not None:
                print(
                    f"     Stability           : mean σ {p.mean_rssi_std:.2f} dB  "
                    f"unstable clicks {p.pct_unstable:.0f}%"
                )
        if p.snr_mean is not None:
            print(
                f"     Link health         : SNR {p.snr_mean:.1f} dB  "
                f"TX {p.tx_bitrate_mean or 0:.0f} Mbps  RX {p.rx_bitrate_mean or 0:.0f} Mbps"
            )
        if p.neighbor_count_same_channel_mean is not None:
            print(
                f"     Interference proxy  : {p.neighbor_count_same_channel_mean:.1f} co-channel  "
                f"{p.neighbor_count_adjacent_mean or 0:.1f} adjacent"
            )
        if p.valid_count:
            print(
                f"     Point coverage      : {100 - p.pct_below_good:.0f}% ≥ {good_dbm:.0f} dBm  "
                f"({p.pct_below_weak:.0f}% weak)"
            )
        if g.valid_cell_count:
            print(
                f"     Interpolated area   : {g.pct_area_good:.0f}% ≥ {good_dbm:.0f} dBm  "
                f"grid min {g.rssi_grid_min:.1f}  grid p10 {g.rssi_grid_p10:.1f}"
            )
        if sm.rooms:
            worst_room = min(sm.rooms, key=lambda r: r.rssi_min or -999)
            print(
                f"     Weakest room        : {worst_room.room_name} "
                f"(min {worst_room.rssi_min:.1f} dBm)"
            )
        print()

    winner = min(results, key=lambda s: s.rank)
    print("-" * 72)
    print(
        f"Recommendation: place router at «{winner.router_name or winner.router_position_id}» "
        f"(session «{winner.session_id}», score {winner.composite_score:.1f})."
    )
    print("=" * 72 + "\n")


def _metrics_as_rows(results: list[SessionMetrics]) -> list[dict]:
    rows = []
    for sm in sorted(results, key=lambda x: x.rank):
        p, g = sm.points, sm.grid
        rows.append({
            "rank": sm.rank,
            "session_id": sm.session_id,
            "router_position_id": sm.router_position_id,
            "router_name": sm.router_name,
            "composite_score": sm.composite_score,
            "point_count": p.point_count,
            "valid_count": p.valid_count,
            "rssi_mean_dbm": p.rssi_mean,
            "rssi_min_dbm": p.rssi_min,
            "rssi_p10_dbm": p.rssi_p10,
            "mean_rssi_std_db": p.mean_rssi_std,
            "snr_mean_db": p.snr_mean,
            "tx_bitrate_mean_mbps": p.tx_bitrate_mean,
            "rx_bitrate_mean_mbps": p.rx_bitrate_mean,
            "neighbor_count_same_channel_mean": p.neighbor_count_same_channel_mean,
            "pct_area_good": g.pct_area_good,
            "pct_area_weak": g.pct_area_weak,
            "rssi_grid_min_dbm": g.rssi_grid_min,
            "rssi_grid_p10_dbm": g.rssi_grid_p10,
        })
    return rows


def export_per_session_heatmaps(
    project_dir: Path,
    results: list[SessionMetrics],
    output_dir: Path,
    *,
    weak_dbm: float,
):
    paths = project_paths(project_dir)
    rooms = _load_rooms(paths["rooms_json"])
    routers = _load_routers(paths["router_positions_json"])
    img = Image.open(paths["floorplan_png"]).convert("RGB")
    img_arr = np.array(img)
    sub = output_dir / "heatmaps"
    sub.mkdir(parents=True, exist_ok=True)

    for sm in results:
        summary = paths["survey_sessions_dir"] / sm.session_id / "measurements_summary.csv"
        measurements = _load_measurements(summary, session_id=sm.session_id)
        if not measurements:
            continue
        prefix = sm.session_id
        highlight = sm.router_position_id or None
        generate_survey_points(
            measurements, img_arr, rooms, routers, sub / f"{prefix}_points.png",
            highlight_position_id=highlight,
        )
        generate_heatmap(
            measurements, img_arr, rooms, routers, sub / f"{prefix}_heatmap.png",
            highlight_position_id=highlight,
        )
        generate_weak_zones(
            measurements,
            img_arr,
            rooms,
            routers,
            sub / f"{prefix}_weak_zones.png",
            threshold_dbm=weak_dbm,
            highlight_position_id=highlight,
        )


def run_session_comparison(
    project_dir: Path,
    session_ids: list[str],
    output_dir: Path,
    *,
    good_threshold: float = DEFAULT_GOOD_DBM,
    weak_threshold: float = DEFAULT_WEAK_DBM,
    export_heatmaps: bool = False,
) -> dict:
    paths = project_paths(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not session_ids:
        session_ids = discover_sessions(paths["survey_sessions_dir"])
    if not session_ids:
        raise ValueError(f"No survey sessions found under {paths['survey_sessions_dir']}")

    results = analyze_project(
        project_dir,
        session_ids,
        good_dbm=good_threshold,
        weak_dbm=weak_threshold,
    )
    valid_results = [sm for sm in results if sm.points.valid_count > 0]
    if not valid_results:
        raise ValueError("No valid measurement data in any session.")

    artifacts = {
        "metrics_csv": output_dir / "comparison_metrics.csv",
        "room_csv": output_dir / "comparison_by_room.csv",
        "bars_png": output_dir / "comparison_bars.png",
        "ranking_png": output_dir / "comparison_ranking.png",
        "best_vs_worst_png": output_dir / "comparison_best_vs_worst.png",
    }
    write_metrics_csv(valid_results, artifacts["metrics_csv"])
    write_room_csv(valid_results, artifacts["room_csv"])
    plot_comparison_bars(valid_results, artifacts["bars_png"])
    plot_ranking(valid_results, artifacts["ranking_png"])

    walk_pairs = build_walk_pairs(project_dir, [sm.session_id for sm in valid_results])
    if walk_pairs:
        artifacts["walk_pairs_csv"] = output_dir / "walk_pairs.csv"
        artifacts["walk_deltas_png"] = output_dir / "comparison_matched_walk_deltas.png"
        write_walk_pairs(walk_pairs, artifacts["walk_pairs_csv"])
        plot_walk_deltas(walk_pairs, artifacts["walk_deltas_png"])

    rooms = _load_rooms(paths["rooms_json"])
    plot_coverage_diff(
        valid_results,
        project_dir,
        rooms,
        artifacts["best_vs_worst_png"],
        good_dbm=good_threshold,
    )

    routers = _load_routers(paths["router_positions_json"])
    artifacts["router_trial_map_png"] = output_dir / "comparison_router_trial_map.png"
    plot_router_trial_map(
        paths["floorplan_png"],
        routers,
        valid_results,
        artifacts["router_trial_map_png"],
    )

    if export_heatmaps:
        export_per_session_heatmaps(
            project_dir,
            valid_results,
            output_dir,
            weak_dbm=weak_threshold,
        )
        artifacts["heatmaps_dir"] = output_dir / "heatmaps"

    return {
        "results": valid_results,
        "metrics": _metrics_as_rows(valid_results),
        "artifacts": artifacts,
        "walk_pairs": walk_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Wi-Fi survey sessions across router placement trials"
    )
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument(
        "--sessions",
        default="",
        help="Comma-separated session IDs (default: all sessions with data)",
    )
    parser.add_argument("--output-dir", default="output/comparison", help="Output directory")
    parser.add_argument(
        "--good-threshold",
        type=float,
        default=DEFAULT_GOOD_DBM,
        help="dBm threshold for 'good' coverage (default −67)",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=DEFAULT_WEAK_DBM,
        help="dBm threshold for 'weak' coverage (default −70)",
    )
    parser.add_argument(
        "--export-heatmaps",
        action="store_true",
        help="Also write per-session heatmap PNGs under output-dir/heatmaps/",
    )
    args = parser.parse_args()

    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    session_ids = (
        [s.strip() for s in args.sessions.split(",") if s.strip()]
        if args.sessions.strip()
        else discover_sessions(paths["survey_sessions_dir"])
    )
    try:
        comparison = run_session_comparison(
            project_dir,
            session_ids,
            Path(args.output_dir),
            good_threshold=args.good_threshold,
            weak_threshold=args.weak_threshold,
            export_heatmaps=args.export_heatmaps,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    print_report(
        comparison["results"],
        good_dbm=args.good_threshold,
        weak_dbm=args.weak_threshold,
    )


if __name__ == "__main__":
    main()
