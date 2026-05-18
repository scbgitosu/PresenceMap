"""
Streamlit app: import and prepare a floorplan image.

Usage:
    streamlit run mac_app/floorplan/floorplan_import.py -- --project data/survey_projects/apartment_test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from mac_app.floorplan import _compat  # noqa: F401 - patch Streamlit before drawable canvas

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from mac_app.floorplan._cli import parse_streamlit_project_args
from shared.project import project_paths
from shared.utils import now_iso

# --- Argument parsing (Streamlit forwards args after `--`, and often `--project` without `--`) ---
def _parse_args():
    return parse_streamlit_project_args()


def _extract_scale_line(canvas_data: dict, sx: float, sy: float) -> dict | None:
    objects = (canvas_data or {}).get("objects", [])
    for obj in reversed(objects):
        if obj.get("type") == "line":
            x1 = float(obj.get("left", 0)) + float(obj.get("x1", 0))
            y1 = float(obj.get("top", 0)) + float(obj.get("y1", 0))
            x2 = float(obj.get("left", 0)) + float(obj.get("x2", 0))
            y2 = float(obj.get("top", 0)) + float(obj.get("y2", 0))
            return {"x1": x1 * sx, "y1": y1 * sy, "x2": x2 * sx, "y2": y2 * sy}
        if obj.get("type") == "path":
            points = []
            for seg in obj.get("path", []):
                if seg[0] in ("M", "L") and len(seg) >= 3:
                    points.append([seg[1], seg[2]])
            if len(points) >= 2:
                (x1, y1), (x2, y2) = points[0], points[-1]
                return {"x1": x1 * sx, "y1": y1 * sy, "x2": x2 * sx, "y2": y2 * sy}
    return None


def main():
    args = _parse_args()
    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    paths["floorplan_metadata"].parent.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="Floorplan Import", layout="wide")
    st.title("Floorplan Import")
    st.caption(f"Project: `{project_dir}`")

    raw_dir = Path("floorplans/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- File selection ---
    raw_images = sorted(raw_dir.glob("*"))
    img_options = [f.name for f in raw_images if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Select source image")
        if img_options:
            selected = st.selectbox("Image in floorplans/raw/", img_options)
            src_path = raw_dir / selected
        else:
            st.info("No images found in `floorplans/raw/`. Upload one below.")
            uploaded = st.file_uploader("Upload floorplan image", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded:
                src_path = raw_dir / uploaded.name
                with open(src_path, "wb") as f:
                    f.write(uploaded.read())
                st.success(f"Saved to {src_path}")
                st.rerun()
            else:
                st.stop()

        img = Image.open(src_path)
        w0, h0 = img.size
        st.write(f"Original size: {w0} × {h0} px")

        st.subheader("2. Rotation")
        rotation = st.selectbox("Rotate", [0, 90, 180, 270], format_func=lambda x: f"{x}°")

        st.subheader("3. Crop (pixels from original)")
        left = st.number_input("Left", 0, w0 - 1, 0, step=1)
        top = st.number_input("Top", 0, h0 - 1, 0, step=1)
        right = st.number_input("Right", 1, w0, w0, step=1)
        bottom = st.number_input("Bottom", 1, h0, h0, step=1)

        st.subheader("4. Scale from a known wall")
        known_dist_ft = st.number_input("Known wall length (feet)", 0.0, 1000.0, 0.0, step=0.5)
        known_dist_px = st.number_input("Fallback: wall length in pixels", 0.0, 10000.0, 0.0, step=1.0)

        notes = st.text_area("Notes (optional)")

    with col2:
        st.subheader("Preview")
        # Apply operations for preview
        preview = img.copy()
        if rotation:
            preview = preview.rotate(-rotation, expand=True)
        if left < right and top < bottom:
            preview = preview.crop((left, top, right, bottom))
        st.image(preview, caption="Processed preview", use_column_width=True)

        new_w, new_h = preview.size
        st.write(f"Output size: {new_w} × {new_h} px")

        max_canvas_w = 1100
        canvas_scale = min(1.0, max_canvas_w / max(new_w, 1))
        canvas_w = max(1, int(round(new_w * canvas_scale)))
        canvas_h = max(1, int(round(new_h * canvas_scale)))
        sx = new_w / canvas_w
        sy = new_h / canvas_h
        st.subheader("Measure one wall")
        st.caption("Draw a single line over a wall with known length, then enter that length in feet.")
        scale_canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#00A0FF",
            background_image=preview,
            update_streamlit=True,
            width=canvas_w,
            height=canvas_h,
            drawing_mode="line",
            key="scale_line_canvas",
        )
        scale_reference = _extract_scale_line(scale_canvas.json_data, sx, sy)
        measured_dist_px = None
        if scale_reference:
            dx = scale_reference["x2"] - scale_reference["x1"]
            dy = scale_reference["y2"] - scale_reference["y1"]
            measured_dist_px = (dx * dx + dy * dy) ** 0.5
            st.write(f"Measured wall length: **{measured_dist_px:.1f} px**")

        scale_ppf = None
        effective_dist_px = measured_dist_px or (known_dist_px if known_dist_px > 0 else None)
        if known_dist_ft > 0 and effective_dist_px:
            scale_ppf = effective_dist_px / known_dist_ft
            st.write(f"Scale: **{scale_ppf:.2f} px/ft**")
            if scale_reference:
                scale_reference["length_ft"] = known_dist_ft

        st.subheader("5. Save")
        if st.button("Save floorplan.png + metadata", type="primary"):
            out_dir = Path("floorplans/processed")
            out_dir.mkdir(parents=True, exist_ok=True)

            processed = img.copy()
            if rotation:
                processed = processed.rotate(-rotation, expand=True)
            if left < right and top < bottom:
                processed = processed.crop((left, top, right, bottom))

            out_path = paths["floorplan_png"]
            processed.save(str(out_path))

            metadata = {
                "image_width_px": new_w,
                "image_height_px": new_h,
                "units": "feet",
                "scale_pixels_per_foot": scale_ppf,
                "scale_reference": scale_reference,
                "origin": {"x": 0, "y": 0},
                "source": str(src_path),
                "rotation_applied": rotation,
                "crop_applied": [left, top, right, bottom] if (left > 0 or top > 0 or right < w0 or bottom < h0) else None,
                "notes": notes,
                "created_at": now_iso(),
            }
            with open(paths["floorplan_metadata"], "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            st.success(f"Saved to `{out_path}` and `{paths['floorplan_metadata']}`")
            st.json(metadata)


if __name__ == "__main__":
    main()
