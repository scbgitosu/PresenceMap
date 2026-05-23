"""Training → Train page: build dataset, train, save to registry."""
from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from mac_app.train.dataset import list_sessions
from mac_app.train.features import DatasetSpec, build_dataset
from mac_app.train.train import TrainConfig, train_session


def render(project_dir: Path) -> None:
    st.header("Training → Train")
    sessions = list_sessions(project_dir)
    if not sessions:
        st.info("No sessions yet. Record labeled data on **ESP32 Nodes** or via `presence-mac esp32-record`.")
        return
    picked: List[str] = st.multiselect("sessions", sessions, default=sessions[:1])
    cols = st.columns(4)
    epochs = cols[0].number_input("epochs", value=30, min_value=1, max_value=500)
    batch = cols[1].number_input("batch size", value=64, min_value=4, max_value=2048)
    val_frac = cols[2].slider("val fraction", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    seq_T = cols[3].number_input("sequence length T", value=8, min_value=2, max_value=32)

    if "train_history" not in st.session_state:
        st.session_state["train_history"] = None
        st.session_state["train_meta"] = None

    if st.button("Build dataset + train", type="primary", disabled=not picked):
        try:
            spec = DatasetSpec(sequence_length=int(seq_T))
            # Validate dataset shape first so we can show a clear error.
            ds = build_dataset(project_dir, picked, spec=spec)
            st.info(f"dataset: {ds.X.shape[0]} sequences, {ds.X.shape[1]} timesteps, {ds.X.shape[2]} features")
            cfg = TrainConfig(epochs=int(epochs), batch_size=int(batch), val_fraction=float(val_frac))
            progress = st.progress(0, text=f"training {int(epochs)} epochs...")
            history_pane = st.empty()

            def _on_epoch(epoch, history):
                progress.progress(epoch / int(epochs), text=f"epoch {epoch}/{int(epochs)}")
                history_pane.line_chart({
                    "train_acc": history.train_acc,
                    "val_acc": history.val_acc,
                })

            meta = train_session(project_dir, picked, spec=spec, cfg=cfg, on_epoch=_on_epoch)
            progress.empty()
            st.session_state["train_meta"] = meta
            st.success(f"saved {meta.model_id}")
        except Exception as exc:
            st.error(f"training failed: {exc}")

    meta = st.session_state.get("train_meta")
    if meta is not None:
        st.subheader("Last trained model")
        st.caption(meta.model_id)
        summary = meta.eval_summary
        cols = st.columns(4)
        cols[0].metric("accuracy", f"{summary['accuracy']:.3f}")
        cols[1].metric("precision", f"{summary['precision']:.3f}")
        cols[2].metric("recall", f"{summary['recall']:.3f}")
        cols[3].metric("f1", f"{summary['f1']:.3f}")
        st.caption(f"confusion (rows=true, cols=pred): {summary['confusion']}")
