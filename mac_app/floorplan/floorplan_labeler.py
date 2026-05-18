"""
Streamlit app: label rooms and place router positions on a floorplan.

Usage:
    streamlit run mac_app/floorplan/floorplan_labeler.py -- --project data/survey_projects/apartment_test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from mac_app.floorplan import _compat  # noqa: F401 — patch Streamlit before drawable canvas

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import patheffects
from matplotlib.figure import Figure
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from mac_app.floorplan._cli import parse_streamlit_project_args
from shared.project import project_paths
from shared.utils import now_iso

# Cap on-canvas display size so large floorplan PNGs fit the browser (coordinates are
# mapped back to full image pixel space when saving).
_MAX_DISPLAY_CANVAS_W = 1200
_MAX_DISPLAY_CANVAS_H = 850


def _display_canvas_dimensions(img_w: int, img_h: int) -> tuple[int, int, float, float]:
    """Return canvas (width, height) and scale factors canvas px → image px."""
    if img_w < 1 or img_h < 1:
        return 1, 1, 1.0, 1.0
    scale = min(_MAX_DISPLAY_CANVAS_W / img_w, _MAX_DISPLAY_CANVAS_H / img_h)
    scale = min(scale, 1.0)
    cw = max(1, int(round(img_w * scale)))
    ch = max(1, int(round(img_h * scale)))
    return cw, ch, img_w / cw, img_h / ch


def _canvas_to_image_polygon(
    poly: List[List[float]], sx: float, sy: float
) -> List[List[float]]:
    return [[p[0] * sx, p[1] * sy] for p in poly]


def _parse_args():
    return parse_streamlit_project_args()


def _polygon_centroid(points: List[List[float]]) -> tuple:
    if not points:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _scale_preview_and_polygons(
    img: Image.Image,
    polygons: List[List[List[float]]],
    max_side: int = 1400,
) -> tuple[Image.Image, List[List[List[float]]]]:
    """Downscale image and polygon coordinates together for a fast matplotlib preview."""
    w, h = img.size
    if w < 1 or h < 1 or not polygons:
        return img, polygons
    if max(w, h) <= max_side:
        return img, polygons
    scale = max_side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    pic = img.resize((nw, nh), Image.Resampling.LANCZOS)
    scaled = [[[p[0] * scale, p[1] * scale] for p in poly] for poly in polygons]
    return pic, scaled


def _room_number_map_figure(
    img: Image.Image,
    polygons: List[List[List[float]]],
) -> Figure:
    """Floorplan with polygons outlined, filled, and labeled 1, 2, 3, … (matches form order)."""
    preview_img, polys = _scale_preview_and_polygons(img, polygons)
    n = len([p for p in polys if len(p) >= 3])
    fig_w = min(14.0, max(6.0, preview_img.width / 100))
    fig_h = min(12.0, max(5.0, preview_img.height / 100))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(preview_img)
    cmap = plt.cm.tab10(np.linspace(0, 1, min(10, max(n, 1))))
    for i, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        c = cmap[i % len(cmap)]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, ys, facecolor=c, alpha=0.28, edgecolor=c, linewidth=2.2)
        cx, cy = _polygon_centroid(poly)
        outline = [patheffects.withStroke(linewidth=3.5, foreground="black")]
        ax.text(
            cx,
            cy,
            str(i + 1),
            fontsize=17,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            path_effects=outline,
        )
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    return fig


def main():
    args = _parse_args()
    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="Floorplan Labeler", layout="wide")
    st.title("Floorplan Labeler")
    st.caption(f"Project: `{project_dir}`")

    floorplan_path = paths["floorplan_png"]
    if not floorplan_path.exists():
        st.error(
            f"`{floorplan_path}` not found. Run **floorplan_import.py** first to generate a floorplan."
        )
        st.stop()

    img = Image.open(floorplan_path)
    img_w, img_h = img.size
    canvas_w, canvas_h, sx, sy = _display_canvas_dimensions(img_w, img_h)

    # Load metadata for scale info
    metadata = {}
    if paths["floorplan_metadata"].exists():
        with open(paths["floorplan_metadata"], encoding="utf-8") as f:
            metadata = json.load(f)

    # Load existing data
    existing_rooms = []
    if paths["rooms_json"].exists():
        with open(paths["rooms_json"], encoding="utf-8") as f:
            existing_rooms = json.load(f)

    existing_routers = []
    if paths["router_positions_json"].exists():
        with open(paths["router_positions_json"], encoding="utf-8") as f:
            existing_routers = json.load(f)

    existing_waypoints = []
    if paths["walk_waypoints_json"].exists():
        with open(paths["walk_waypoints_json"], encoding="utf-8") as f:
            existing_waypoints = json.load(f)

    tab_rooms, tab_routers, tab_walk, tab_config = st.tabs([
        "Rooms",
        "Router Positions",
        "Walk Template",
        "Project Config",
    ])

    # ---- Rooms tab ----
    with tab_rooms:
        st.subheader("Draw room polygons")
        st.info(
            "**How to draw:** **Left-click** = add a corner · **Right-click** = finish that room · "
            "**Double-click** = remove the last corner (not “done”). "
            "When **Numbered map** appears below, each closed region is labeled **1, 2, 3…** — "
            "those numbers are the same **Room 1, Room 2, …** in the form."
        )
        if canvas_h < img_h or canvas_w < img_w:
            st.caption(
                f"Display scaled to **{canvas_w}×{canvas_h}** px to fit the window "
                f"(floorplan is {img_w}×{img_h} px). Saved shapes use **full-resolution** coordinates."
            )

        canvas_result = st_canvas(
            fill_color="rgba(100, 100, 200, 0.15)",
            stroke_width=2,
            stroke_color="#6464C8",
            background_image=img,
            update_streamlit=True,
            width=canvas_w,
            height=canvas_h,
            drawing_mode="polygon",
            key="rooms_canvas",
        )

        polygons = []
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            for obj in canvas_result.json_data["objects"]:
                if obj.get("type") == "path":
                    # streamlit-drawable-canvas returns polygon paths as SVG path segments
                    points = []
                    path = obj.get("path", [])
                    for seg in path:
                        if seg[0] in ("M", "L") and len(seg) >= 3:
                            points.append([seg[1], seg[2]])
                    if len(points) >= 3:
                        polygons.append(
                            _canvas_to_image_polygon(points, sx, sy),
                        )

        if polygons:
            display_polys = polygons
        else:
            display_polys = []
            for r in existing_rooms:
                p = r.get("polygon")
                if p and len(p) >= 3:
                    display_polys.append(p)

        st.write(
            f"**{len(display_polys)}** room outline(s) to name — "
            + (
                "from the canvas above."
                if polygons
                else "from `rooms.json` (canvas is empty; draw new shapes to replace)."
            )
        )

        if display_polys:
            st.subheader("Numbered map")
            st.caption(
                "Each region’s **color and number** match **Room 1 / Room 2 / …** below. "
                "Order is usually the order polygons were finished on the canvas; "
                "if only a saved file is loaded, it follows the order in that file."
            )
            fig = _room_number_map_figure(img, display_polys)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.subheader("Name each room")
            room_rows = []
            for i, poly in enumerate(display_polys):
                cx, cy = _polygon_centroid(poly)
                existing = existing_rooms[i] if i < len(existing_rooms) else {}
                st.markdown(
                    f"##### Room {i + 1} — {len(poly)} corners · label centre ≈ ({cx:.0f}, {cy:.0f}) px"
                )
                col1, col2 = st.columns(2)
                with col1:
                    rid = st.text_input(
                        "Short ID (stored in JSON)",
                        value=existing.get("room_id", f"room_{i + 1}"),
                        placeholder="kitchen, bed_1, bath…",
                        key=f"room_id_{i}",
                    )
                with col2:
                    rname = st.text_input(
                        "Display name",
                        value=existing.get("room_name", ""),
                        placeholder="Kitchen, Primary bedroom…",
                        key=f"room_name_{i}",
                    )
                room_rows.append({
                    "room_id": rid,
                    "room_name": rname,
                    "polygon": poly,
                    "label_x": cx,
                    "label_y": cy,
                })

            if st.button("Save rooms.json", type="primary", key="save_rooms_json"):
                with open(paths["rooms_json"], "w", encoding="utf-8") as f:
                    json.dump(room_rows, f, indent=2)
                st.success(f"Saved {len(room_rows)} rooms to `{paths['rooms_json']}`")
                st.json(room_rows)
        else:
            st.info(
                "Close at least one polygon on the canvas (right-click to finish), "
                "or ensure `rooms.json` contains polygons — then a numbered map and naming fields appear."
            )

        if existing_rooms:
            st.subheader("Currently saved rooms.json")
            st.json(existing_rooms)

    # ---- Router positions tab ----
    with tab_routers:
        st.subheader("Place router positions")
        st.info("Select **Point** mode, then click to place router/AP positions.")
        if canvas_h < img_h or canvas_w < img_w:
            st.caption(
                f"Same display scale as Rooms (**{canvas_w}×{canvas_h}** px). "
                "Saved positions are in full floorplan pixels."
            )

        router_canvas = st_canvas(
            fill_color="rgba(255, 165, 0, 0.6)",
            stroke_width=2,
            stroke_color="#FFA500",
            background_image=img,
            update_streamlit=True,
            width=canvas_w,
            height=canvas_h,
            drawing_mode="point",
            point_display_radius=8,
            key="router_canvas",
        )

        router_points = []
        if router_canvas.json_data and router_canvas.json_data.get("objects"):
            for obj in router_canvas.json_data["objects"]:
                if obj.get("type") == "circle":
                    rx = float(obj.get("left", 0)) * sx
                    ry = float(obj.get("top", 0)) * sy
                    router_points.append([rx, ry])

        st.write(f"**{len(router_points)} point(s) placed**")

        if router_points:
            st.subheader("Router position metadata")
            router_rows = []
            for i, (px, py) in enumerate(router_points):
                existing = existing_routers[i] if i < len(existing_routers) else {}
                col1, col2, col3 = st.columns(3)
                with col1:
                    rp_id = st.text_input(
                        f"Position ID #{i+1}",
                        value=existing.get("router_position_id", f"pos_{i+1}"),
                        key=f"rp_id_{i}",
                    )
                with col2:
                    rp_name = st.text_input(
                        f"Position name #{i+1}",
                        value=existing.get("name", ""),
                        key=f"rp_name_{i}",
                    )
                with col3:
                    rp_height = st.number_input(
                        f"Height ft #{i+1}",
                        0.0, 20.0,
                        value=float(existing.get("height_ft", 4.0)),
                        step=0.5,
                        key=f"rp_height_{i}",
                    )
                rp_notes = st.text_input(
                    f"Notes #{i+1}",
                    value=existing.get("notes", ""),
                    key=f"rp_notes_{i}",
                )
                router_rows.append({
                    "router_position_id": rp_id,
                    "name": rp_name,
                    "x_px": px,
                    "y_px": py,
                    "height_ft": rp_height,
                    "notes": rp_notes,
                })

            if st.button("Save router_positions.json", type="primary"):
                with open(paths["router_positions_json"], "w", encoding="utf-8") as f:
                    json.dump(router_rows, f, indent=2)
                st.success(f"Saved {len(router_rows)} positions to `{paths['router_positions_json']}`")

        if existing_routers:
            st.subheader("Currently saved router_positions.json")
            st.json(existing_routers)

    # ---- Walk template tab ----
    with tab_walk:
        st.subheader("Create matched walk waypoints")
        st.info(
            "Select Point mode, then click survey locations in the order you want to walk them. "
            "Using the same waypoints across router trials makes session comparisons much stronger."
        )
        waypoint_canvas = st_canvas(
            fill_color="rgba(0, 160, 255, 0.7)",
            stroke_width=2,
            stroke_color="#00A0FF",
            background_image=img,
            update_streamlit=True,
            width=canvas_w,
            height=canvas_h,
            drawing_mode="point",
            point_display_radius=7,
            key="walk_waypoints_canvas",
        )

        waypoint_points = []
        if waypoint_canvas.json_data and waypoint_canvas.json_data.get("objects"):
            for obj in waypoint_canvas.json_data["objects"]:
                if obj.get("type") == "circle":
                    wx = float(obj.get("left", 0)) * sx
                    wy = float(obj.get("top", 0)) * sy
                    waypoint_points.append([wx, wy])

        display_waypoints = waypoint_points or [
            [float(w.get("x_px", 0)), float(w.get("y_px", 0))]
            for w in existing_waypoints
        ]
        st.write(f"**{len(display_waypoints)} waypoint(s)**")
        if display_waypoints:
            waypoint_rows = []
            for i, (px, py) in enumerate(display_waypoints):
                existing = existing_waypoints[i] if i < len(existing_waypoints) else {}
                col1, col2 = st.columns(2)
                with col1:
                    waypoint_id = st.text_input(
                        f"Waypoint ID #{i + 1}",
                        value=existing.get("waypoint_id", f"wp{i + 1:02d}"),
                        key=f"waypoint_id_{i}",
                    )
                with col2:
                    label_value = st.text_input(
                        f"Label #{i + 1}",
                        value=existing.get("label", ""),
                        key=f"waypoint_label_{i}",
                    )
                waypoint_rows.append({
                    "waypoint_id": waypoint_id,
                    "order": i + 1,
                    "x_px": px,
                    "y_px": py,
                    "label": label_value,
                })

            if st.button("Save walk_waypoints.json", type="primary"):
                with open(paths["walk_waypoints_json"], "w", encoding="utf-8") as f:
                    json.dump(waypoint_rows, f, indent=2)
                st.success(f"Saved {len(waypoint_rows)} waypoints to `{paths['walk_waypoints_json']}`")
                st.json(waypoint_rows)

        if existing_waypoints:
            st.subheader("Currently saved walk_waypoints.json")
            st.json(existing_waypoints)

    # ---- Config tab ----
    with tab_config:
        st.subheader("Project configuration")

        existing_cfg = {}
        if paths["project_config"].exists():
            with open(paths["project_config"], encoding="utf-8") as f:
                existing_cfg = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            proj_name = st.text_input(
                "Project name",
                value=existing_cfg.get("project_name", project_dir.name),
            )
            target_ssid = st.text_input(
                "Target SSID",
                value=existing_cfg.get("target_ssid", ""),
            )
            target_bssid = st.text_input(
                "Target BSSID (optional)",
                value=existing_cfg.get("target_bssid", ""),
            )
        with col2:
            default_iface = st.text_input(
                "Default Wi-Fi interface",
                value=existing_cfg.get("default_interface", "wlan0"),
            )
            units = st.selectbox(
                "Units",
                ["feet", "meters"],
                index=0 if existing_cfg.get("units", "feet") == "feet" else 1,
            )
            scan_backend = st.selectbox(
                "Scan backend",
                ["iw", "auto", "nmcli"],
                index=["iw", "auto", "nmcli"].index(existing_cfg.get("scan_backend", "iw"))
                if existing_cfg.get("scan_backend", "iw") in ["iw", "auto", "nmcli"] else 0,
            )

        if st.button("Save project_config.json", type="primary"):
            cfg = {
                "project_name": proj_name,
                "target_ssid": target_ssid,
                "target_bssid": target_bssid,
                "default_interface": default_iface,
                "units": units,
                "collection_mode": "click_to_scan",
                "scan_backend": scan_backend,
                "paths": {
                    "floorplan_png": str(paths["floorplan_png"]),
                    "rooms_json": str(paths["rooms_json"]),
                    "router_positions_json": str(paths["router_positions_json"]),
                    "walk_waypoints_json": str(paths["walk_waypoints_json"]),
                    "survey_sessions_dir": str(paths["survey_sessions_dir"]),
                },
                "updated_at": now_iso(),
            }
            with open(paths["project_config"], "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            st.success(f"Saved to `{paths['project_config']}`")
            st.json(cfg)

        if existing_cfg:
            st.subheader("Current project_config.json")
            st.json(existing_cfg)


if __name__ == "__main__":
    main()
