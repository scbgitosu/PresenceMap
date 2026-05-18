"""Restore ``streamlit.elements.image.image_to_url`` for streamlit-drawable-canvas.

Modern Streamlit (1.41+) removed this helper from the public image module; the
canvas still calls it with the legacy (width-as-int) signature. We delegate to
``image_utils.image_to_url`` with a ``LayoutConfig`` built from that width.

Imported for side effects at the top of the floorplan apps.
"""
from __future__ import annotations

import streamlit.elements.image as st_image_module
from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_impl
from streamlit.elements.lib.layout_utils import create_layout_config


def _image_to_url_legacy(
    image: object,
    width: int,
    clamp: bool,
    channels: str,
    output_format: str,
    image_id: str,
) -> str:
    layout_config = create_layout_config(width=width)
    return _image_to_url_impl(
        image, layout_config, clamp, channels, output_format, image_id
    )


def apply_patches() -> None:
    if not hasattr(st_image_module, "image_to_url"):
        st_image_module.image_to_url = _image_to_url_legacy  # type: ignore[attr-defined]


apply_patches()
