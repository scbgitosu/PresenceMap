"""Models page: list registry + compare two models side-by-side."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mac_app.train.registry import list_models, read_eval


def render(project_dir: Path) -> None:
    st.header("Models")
    models = list_models(project_dir)
    if not models:
        st.info("no models trained yet. Use Training → Train to create one.")
        return
    rows = []
    for m in models:
        s = m.eval_summary
        rows.append(
            {
                "model_id": m.model_id,
                "created_at": m.created_at,
                "sessions": ", ".join(m.session_ids),
                "accuracy": s.get("accuracy"),
                "precision": s.get("precision"),
                "recall": s.get("recall"),
                "f1": s.get("f1"),
                "n_params": m.n_params,
                "git_sha": m.code_git_sha,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Compare two")
    ids = [m.model_id for m in models]
    cols = st.columns(2)
    a = cols[0].selectbox("model A", ids, key="cmp_a", index=0)
    b = cols[1].selectbox(
        "model B",
        ids,
        key="cmp_b",
        index=min(1, len(ids) - 1),
    )
    if a == b:
        st.info("pick two different models to compare")
        return
    eval_a = read_eval(project_dir, a)
    eval_b = read_eval(project_dir, b)
    if not eval_a or not eval_b:
        st.error("missing eval.json for one of the selected models")
        return

    cols = st.columns(2)
    with cols[0]:
        st.caption(a)
        st.json(eval_a["summary"])
    with cols[1]:
        st.caption(b)
        st.json(eval_b["summary"])

    # Disagreement timeline.
    pw_a = pd.DataFrame(eval_a.get("per_window", []))
    pw_b = pd.DataFrame(eval_b.get("per_window", []))
    if pw_a.empty or pw_b.empty:
        return
    merged = pw_a.merge(pw_b, on=["window_id", "session_id"], suffixes=("_a", "_b"))
    merged["disagree"] = merged["y_pred_a"] != merged["y_pred_b"]
    st.subheader("Per-window comparison")
    st.dataframe(
        merged[["window_id", "session_id", "y_true_a", "y_pred_a", "p_occupied_a", "y_pred_b", "p_occupied_b", "disagree"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"disagreement count: {int(merged['disagree'].sum())} / {len(merged)}")
