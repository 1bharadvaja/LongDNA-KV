"""Plot LongDNA-KV evaluation outputs."""

from __future__ import annotations

import os
from pathlib import Path

Path("results/.matplotlib").mkdir(parents=True, exist_ok=True)
Path("results/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(results_dir: str = "results") -> None:
    out = Path(results_dir)
    acc_path = out / "accuracy_vs_budget.csv"
    mem_path = out / "memory_vs_length.csv"
    if not acc_path.exists() or not mem_path.exists():
        raise FileNotFoundError("run eval.py before plots.py")

    acc = pd.read_csv(acc_path)
    mem = pd.read_csv(mem_path)

    for seq_len, group in acc.groupby("seq_len"):
        plt.figure(figsize=(8, 5))
        for policy, pgroup in group.groupby("policy"):
            pgroup = pgroup.sort_values("kv_budget")
            plt.plot(pgroup["kv_budget"], pgroup["accuracy"], marker="o", label=policy)
        plt.title(f"Final-token accuracy vs KV budget, length {seq_len}")
        plt.xlabel("KV budget")
        plt.ylabel("Accuracy")
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"accuracy_vs_budget_len{seq_len}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    for policy, group in mem.groupby("policy"):
        group = group.sort_values("seq_len")
        plt.plot(group["seq_len"], group["estimated_kv_bytes"] / (1024 * 1024), marker="o", label=policy)
    plt.title("Estimated KV memory vs context length")
    plt.xlabel("Context length")
    plt.ylabel("KV memory (MiB)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "memory_vs_length.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    make_plots()
    print("wrote plots to results/")
