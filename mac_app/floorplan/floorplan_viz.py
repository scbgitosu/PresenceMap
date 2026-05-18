"""Floorplan overlays for router trials and placement suggestions."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


def draw_trial_routers(
    ax,
    routers: list,
    *,
    highlight_position_id: Optional[str] = None,
    dim_unhighlighted: bool = False,
):
    """Draw labeled orange stars for router/AP candidate positions from labeling."""
    for rp in routers:
        rid = rp.get("router_position_id", "")
        name = rp.get("name") or rid
        x, y = float(rp["x_px"]), float(rp["y_px"])
        is_active = highlight_position_id and rid == highlight_position_id
        if is_active:
            color = "#1565c0"
            edge = "#0d47a1"
            size = 22
            z = 25
        else:
            color = "orange" if not dim_unhighlighted else "#cccccc"
            edge = "black" if not dim_unhighlighted else "#888888"
            size = 16 if not dim_unhighlighted else 12
            z = 20
        ax.plot(x, y, marker="*", markersize=size, color=color, markeredgecolor=edge, zorder=z)
        label = f"★ {name}"
        if is_active:
            label += " (this trial)"
        ax.text(
            x + 10,
            y - 10,
            label,
            fontsize=9,
            fontweight="bold" if is_active else "normal",
            color=edge,
            zorder=z + 1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )


def draw_suggested_candidates(ax, candidates: list):
    """Draw ranked optimizer suggestions (rank 1 = green star, others = numbered circles)."""
    for cand in candidates:
        rank = int(cand.get("rank", 0))
        x, y = float(cand["x_px"]), float(cand["y_px"])
        if rank == 1:
            ax.plot(
                x,
                y,
                marker="*",
                markersize=24,
                color="#00c800",
                markeredgecolor="#006400",
                linewidth=1.2,
                zorder=30,
            )
            ax.text(
                x + 12,
                y + 12,
                f"#1 suggested\n({x:.0f}, {y:.0f})",
                fontsize=9,
                fontweight="bold",
                color="#006400",
                zorder=31,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f5e9", alpha=0.95, edgecolor="#00a050"),
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                markersize=11,
                color="#7b1fa2",
                markeredgecolor="white",
                linewidth=1.5,
                zorder=28,
            )
            ax.text(
                x + 8,
                y - 8,
                f"#{rank}",
                fontsize=8,
                fontweight="bold",
                color="#4a148c",
                zorder=29,
            )


def _legend_for_placement(routers: list, candidates: list) -> list:
    items = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="black",
            markersize=14,
            label="Labeled trial AP (Hall / Fridge / Original)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#00c800",
            markeredgecolor="#006400",
            markersize=16,
            label="#1 suggested new AP",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#7b1fa2",
            markeredgecolor="white",
            markersize=10,
            label="#2–#5 alternate suggestions",
            linestyle="None",
        ),
    ]
    return items


def plot_placement_overview(
    image_path: Path,
    routers: list,
    candidates: list,
    output_path: Path,
    *,
    title: str = "Router trials and suggested placement",
):
    """Clear floorplan-only map: trial positions + ranked suggestions (no heatmap)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(np.array(img), origin="upper", zorder=0)
    draw_trial_routers(ax, routers)
    draw_suggested_candidates(ax, candidates)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.legend(handles=_legend_for_placement(routers, candidates), loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_router_trial_map(
    image_path: Path,
    routers: list,
    session_results: list,
    output_path: Path,
):
    """
    Floorplan showing which labeled AP position each session tested and how it ranked.

    session_results: SessionMetrics-like objects with router_position_id, router_name,
    session_id, rank, composite_score.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    by_router = {sm.router_position_id: sm for sm in session_results}

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(np.array(img), origin="upper", zorder=0)

    rank_colors = {1: "#00a050", 2: "#7cb342", 3: "#ffb300"}
    legend_items = []

    for rp in routers:
        rid = rp.get("router_position_id", "")
        name = rp.get("name") or rid
        x, y = float(rp["x_px"]), float(rp["y_px"])
        sm = by_router.get(rid)
        if sm:
            rank = sm.rank
            color = rank_colors.get(rank, "#888888")
            label = (
                f"#{rank} {name}\n"
                f"session: {sm.session_id}\n"
                f"score: {sm.composite_score:.1f}"
            )
        else:
            color = "#bbbbbb"
            label = f"{name}\n(not surveyed)"
        ax.plot(x, y, marker="*", markersize=20, color=color, markeredgecolor="black", zorder=20)
        ax.text(
            x + 12,
            y - 12,
            label,
            fontsize=8,
            color="black",
            zorder=21,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor=color, linewidth=2),
        )
        if sm:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=14,
                    label=f"#{sm.rank} {name} ({sm.session_id})",
                    linestyle="None",
                )
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", fontsize=8, title="Trial ranking")
    ax.set_title("Which AP location each session measured", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def annotate_placement_heatmap_ax(ax, routers: list, candidates: list):
    """Add trial + suggestion markers on an existing coverage axes."""
    draw_trial_routers(ax, routers, dim_unhighlighted=False)
    draw_suggested_candidates(ax, candidates)
    ax.legend(handles=_legend_for_placement(routers, candidates), loc="lower right", fontsize=7)
