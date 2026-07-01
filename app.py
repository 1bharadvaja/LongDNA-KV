"""Streamlit demo for biology-aware KV retention on synthetic DNA."""

from __future__ import annotations

import os
from pathlib import Path

Path("results/.matplotlib").mkdir(parents=True, exist_ok=True)
Path("results/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from data import ITOS, MOTIF_TO_TARGET, TASKS, generate_example, motif_window
from eval import DEFAULT_POLICIES, load_model, oracle_attention_scores, pick_device, predict_incremental, predict_simulated
from kv_policies import POLICIES, estimate_kv_bytes
from real_data import parse_fasta, sample_fasta_window


def plot_retained(seq_len: int, retained: list[int], ex):
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.hlines(0, 0, seq_len - 1, color="#c8ccd4", linewidth=8)
    if retained:
        ax.scatter(retained, [0] * len(retained), s=16, color="#2563eb", label="retained KV", zorder=3)
    motif_positions = motif_window(ex)
    ax.axvspan(min(motif_positions), max(motif_positions), color="#16a34a", alpha=0.25, label="motif neighborhood")
    for pos in ex.distractor_positions:
        ax.axvspan(pos, pos + 8, color="#dc2626", alpha=0.18, label="distractor")
    if ex.paired_motif_start is not None:
        ax.axvspan(ex.paired_motif_start, ex.paired_motif_start + 8, color="#7c3aed", alpha=0.20, label="paired motif")
    if ex.old_region_start is not None and ex.old_region_end is not None:
        ax.axvspan(ex.old_region_start, ex.old_region_end, color="#0891b2", alpha=0.20, label="old GC region")
    ax.axvspan(ex.query_start, seq_len - 1, color="#f97316", alpha=0.22, label="query")
    ax.set_xlim(0, seq_len - 1)
    ax.set_yticks([])
    ax.set_xlabel("DNA prompt position")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.45), fontsize=8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    return fig


@st.cache_resource
def cached_model(checkpoint: str, device_name: str):
    device = pick_device(device_name)
    return load_model(checkpoint, device), device


def main() -> None:
    st.set_page_config(page_title="LongDNA-KV", layout="wide")
    st.title("LongDNA-KV")

    with st.sidebar:
        checkpoint = st.text_input("Checkpoint", value="results/model.pt")
        seq_len = st.select_slider("Sequence length", options=[512, 1024, 2048, 4096, 8192], value=1024)
        budget = st.select_slider("KV budget", options=[64, 128, 256, 512, 1024], value=256)
        policy = st.selectbox("Retention policy", POLICIES, index=4)
        task = st.selectbox("Probe task", TASKS, index=0)
        background = st.radio("Background", ["random", "fasta"], index=0)
        fasta_path = st.text_input("FASTA path", value="data/example.fa")
        mode = st.radio("Inference mode", ["simulation", "incremental"], index=0)
        include_oracle = st.checkbox("Include oracle in comparison", value=False)
        seed = st.number_input("Example seed", min_value=0, max_value=999999, value=13, step=1)

    model, device = cached_model(checkpoint, "auto")
    if not Path(checkpoint).exists():
        st.warning(f"No {checkpoint} checkpoint found. Run `python train.py` for meaningful predictions.")

    import random

    rng = random.Random(int(seed))
    bg_seq, bg_meta = None, None
    if background == "fasta":
        try:
            records = parse_fasta(fasta_path)
            bg_seq, bg_meta = sample_fasta_window(records, seq_len, rng=rng)
        except Exception as exc:
            st.warning(f"Could not load FASTA background: {exc}. Falling back to random background.")
            background = "random"
    ex = generate_example(
        seq_len=seq_len,
        rng=rng,
        task=task,
        background=background,
        background_sequence=bg_seq,
        background_metadata=bg_meta,
    )
    oracle = oracle_attention_scores(model, ex.prompt_ids[None, :].to(device)) if policy == "attention_oracle" else None
    predictor = predict_incremental if mode == "incremental" else predict_simulated
    result = predictor(model, ex.prompt_ids, policy, budget, device, oracle)
    pred = ITOS[result["pred_id"]]
    correct = pred == ex.target

    cols = st.columns(5)
    cols[0].metric("Early motif", ex.motif)
    cols[1].metric("True target", ex.target)
    cols[2].metric("Prediction", pred)
    cols[3].metric("Correct", "yes" if correct else "no")
    cols[4].metric("Retained KV", len(result["retained"]))

    kv_bytes = estimate_kv_bytes(len(result["retained"]), model.cfg.n_layers, model.cfg.n_heads, model.head_dim)
    full_bytes = estimate_kv_bytes(seq_len - 1, model.cfg.n_layers, model.cfg.n_heads, model.head_dim)
    st.caption(
        f"Estimated KV memory: {kv_bytes / (1024 * 1024):.2f} MiB "
        f"({full_bytes / max(kv_bytes, 1):.1f}x smaller than full cache)"
    )

    left, right = st.columns([2, 1])
    with left:
        st.pyplot(plot_retained(seq_len, result["retained"], ex), clear_figure=True)
    with right:
        st.write(
            {
                "task": ex.task_name,
                "background": ex.background_source,
                "record_id": ex.record_id,
                "window": None if ex.window_start is None else f"{ex.window_start}-{ex.window_end}",
                "gc_content": None if ex.gc_content is None else round(ex.gc_content, 3),
            }
        )
        st.text_area("Query region", ex.sequence[ex.query_start :], height=80)
        motif_context = ex.sequence[max(0, ex.motif_start - 8) : ex.motif_start + len(ex.motif) + 8]
        st.text_area("Motif context", motif_context, height=80)
        if ex.distractor_positions:
            st.caption(f"Distractors: {list(zip(ex.distractor_motifs, ex.distractor_positions))}")
        if ex.paired_motif_start is not None:
            st.caption(f"Paired motif: {ex.paired_motif} at {ex.paired_motif_start}")
        if ex.old_region_start is not None:
            st.caption(f"Old GC region: {ex.old_region_start}-{ex.old_region_end}")

    rows = []
    comparison_policies = list(DEFAULT_POLICIES)
    if include_oracle:
        comparison_policies.append("attention_oracle")
    for p in comparison_policies:
        p_oracle = oracle_attention_scores(model, ex.prompt_ids[None, :].to(device)) if p == "attention_oracle" else None
        p_result = predict_simulated(model, ex.prompt_ids, p, budget, device, p_oracle)
        p_pred = ITOS[p_result["pred_id"]]
        rows.append(
            {
                "policy": p,
                "prediction": p_pred,
                "correct": p_pred == ex.target,
                "retained_kv": len(p_result["retained"]),
                "motif_positions_retained": len(set(p_result["retained"]).intersection(set(motif_window(ex)))),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Synthetic mapping"):
        st.table(pd.DataFrame([{"motif": k, "target": v} for k, v in MOTIF_TO_TARGET.items()]))


if __name__ == "__main__":
    main()
