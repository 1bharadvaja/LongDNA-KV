"""Evaluate KV retention policies for the LongDNA-KV demo."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd
import torch

from data import ITOS, TASKS, generate_example
from kv_policies import POLICIES, estimate_kv_bytes, select_retained_positions
from model import ModelConfig, TinyDNATransformer, prune_kv_cache
from real_data import parse_fasta, sample_fasta_window


DEFAULT_POLICIES = [p for p in POLICIES if p != "attention_oracle"]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device) -> TinyDNATransformer:
    if Path(path).exists():
        ckpt = torch.load(path, map_location=device)
        cfg = ModelConfig.from_dict(ckpt["model_config"])
        model = TinyDNATransformer(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.train_args = ckpt.get("train_args", {})
    else:
        print(f"warning: {path} not found; evaluating an untrained model")
        model = TinyDNATransformer(ModelConfig())
        model.train_args = {}
    return model.to(device).eval()


def final_step_attention_mask(seq_len: int, retained: Sequence[int], device: torch.device) -> torch.Tensor:
    allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    allowed[-1, :] = False
    if retained:
        allowed[-1, torch.tensor(list(retained), dtype=torch.long, device=device)] = True
    allowed[-1, -1] = True
    return allowed[None, :, :]


@torch.no_grad()
def oracle_attention_scores(model: TinyDNATransformer, prompt: torch.Tensor) -> Dict[int, float]:
    out = model(prompt, output_attentions=True)
    if not out["attentions"]:
        return {}
    # Last layer, final query, mean over heads.
    attn = out["attentions"][-1][0, :, -1, :].mean(dim=0).detach().cpu()
    return {i: float(attn[i]) for i in range(prompt.size(1) - 1)}


@torch.no_grad()
def predict_simulated(
    model: TinyDNATransformer,
    prompt_ids: torch.Tensor,
    policy: str,
    budget: int,
    device: torch.device,
    oracle_scores_map: Mapping[int, float] | None = None,
) -> dict:
    prompt = prompt_ids[None, :].to(device)
    seq_len = prompt.size(1)
    tokens = [int(x) for x in prompt_ids.tolist()]
    retained = select_retained_positions(
        tokens,
        upto=seq_len - 1,
        budget=budget,
        policy=policy,
        oracle_scores=oracle_scores_map,
    )
    mask = final_step_attention_mask(seq_len, retained, device)
    out = model(prompt, attention_mask=mask)
    logits = out["logits"][:, -1, :]
    return {"pred_id": int(logits.argmax(dim=-1).item()), "retained": retained}


@torch.no_grad()
def predict_incremental(
    model: TinyDNATransformer,
    prompt_ids: torch.Tensor,
    policy: str,
    budget: int,
    device: torch.device,
    oracle_scores_map: Mapping[int, float] | None = None,
) -> dict:
    tokens = [int(x) for x in prompt_ids.tolist()]
    cache = None
    cache_positions: List[int] = []
    logits = None
    used_for_final: List[int] = []
    for pos, token_id in enumerate(tokens):
        if pos == len(tokens) - 1:
            used_for_final = list(cache_positions)
        x = torch.tensor([[token_id]], dtype=torch.long, device=device)
        position_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
        out = model(x, past_key_values=cache, use_cache=True, position_ids=position_ids)
        logits = out["logits"][:, -1, :]
        cache = out["past_key_values"]
        cache_positions = [*cache_positions, pos]

        # The policy decision is an explicit eviction/retention step over old
        # K/V slots. No summary vector is introduced; retained slots stay as K/V.
        retained = select_retained_positions(
            tokens,
            upto=pos + 1,
            budget=budget,
            policy=policy,
            oracle_scores=oracle_scores_map,
        )
        cache = prune_kv_cache(cache, cache_positions, retained)
        cache_positions = retained

    assert logits is not None
    return {"pred_id": int(logits.argmax(dim=-1).item()), "retained": used_for_final}


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    train_args = getattr(model, "train_args", {})
    train_seq_len = train_args.get("seq_len")
    if train_seq_len is not None:
        print(f"Checkpoint training seq_len: {train_seq_len}")
    else:
        print("Checkpoint training seq_len: unknown")
    print(f"Eval seq_lengths: {args.seq_lengths}")
    if train_seq_len is not None:
        for seq_len in args.seq_lengths:
            if seq_len != int(train_seq_len):
                print(
                    f"Warning: eval seq_len={seq_len} differs from checkpoint train seq_len={train_seq_len}. "
                    "Accuracy may be out-of-distribution."
                )
    train_task = train_args.get("task")
    if train_task and train_task != args.task:
        print(f"Warning: checkpoint task={train_task!r}, but eval task={args.task!r}.")
    train_background = train_args.get("background")
    if train_background and train_background != args.background:
        print(f"Warning: checkpoint background={train_background!r}, but eval background={args.background!r}.")

    if "attention_oracle" in args.policies and not args.include_oracle:
        print("Warning: dropping attention_oracle because --include-oracle was not set.")
        args.policies = [p for p in args.policies if p != "attention_oracle"]

    fasta_records = None
    if args.background == "fasta":
        if not args.fasta_path:
            raise ValueError("--background fasta requires --fasta-path")
        fasta_records = parse_fasta(args.fasta_path)
        print(f"Loaded {len(fasta_records)} FASTA record(s) from {args.fasta_path}")

    rng = random.Random(args.seed)
    rows = []
    memory_rows = []

    predictor = predict_incremental if args.mode == "incremental" else predict_simulated

    for seq_len in args.seq_lengths:
        examples = []
        for _ in range(args.num_examples):
            if args.background == "fasta":
                bg_seq, bg_meta = sample_fasta_window(fasta_records, seq_len, rng=rng)
            else:
                bg_seq, bg_meta = None, None
            examples.append(
                generate_example(
                    seq_len=seq_len,
                    rng=rng,
                    task=args.task,
                    background=args.background,
                    background_sequence=bg_seq,
                    background_metadata=bg_meta,
                )
            )
        if args.debug_plain:
            print(f"\nDEBUG seq_len={seq_len}")
            for i, ex in enumerate(examples[:5]):
                plain = predict_plain(model, ex.prompt_ids, device)
                sim = predict_simulated(model, ex.prompt_ids, "full_cache", 128, device)["pred_id"]

                print("example", i)
                print("target:", ex.target_id, ITOS[ex.target_id])
                print("plain: ", plain, ITOS[plain])
                print("sim:   ", sim, ITOS[sim])
                print("prompt tail:", "".join(ITOS[int(x)] for x in ex.prompt_ids[-50:]))
                print()
        for budget in args.budgets:
            for policy in args.policies:
                correct = 0
                total_retained = 0
                start = time.perf_counter()
                for ex in examples:
                    oracle = oracle_attention_scores(model, ex.prompt_ids[None, :].to(device)) if policy == "attention_oracle" else None
                    result = predictor(model, ex.prompt_ids, policy, budget, device, oracle)
                    correct += int(result["pred_id"] == ex.target_id)
                    total_retained += len(result["retained"])
                elapsed = time.perf_counter() - start
                avg_retained = total_retained / len(examples)
                kv_bytes = estimate_kv_bytes(
                    int(avg_retained),
                    model.cfg.n_layers,
                    model.cfg.n_heads,
                    model.head_dim,
                    bytes_per_value=args.bytes_per_value,
                )
                full_retained = max(1, seq_len - 1)
                rows.append(
                    {
                        "mode": args.mode,
                        "seq_len": seq_len,
                        "kv_budget": budget,
                        "policy": policy,
                        "accuracy": correct / len(examples),
                        "avg_retained_kv": avg_retained,
                        "estimated_kv_bytes": kv_bytes,
                        "compression_ratio_vs_full": full_retained / max(1.0, avg_retained),
                        "tokens_per_second": (seq_len * len(examples)) / max(elapsed, 1e-9),
                    }
                )

        full_bytes = estimate_kv_bytes(seq_len - 1, model.cfg.n_layers, model.cfg.n_heads, model.head_dim, args.bytes_per_value)
        memory_rows.append({"seq_len": seq_len, "policy": "full_cache", "retained_kv": seq_len - 1, "estimated_kv_bytes": full_bytes})
        for budget in args.budgets:
            retained = min(budget, seq_len - 1)
            memory_rows.append(
                {
                    "seq_len": seq_len,
                    "policy": f"budget_{budget}",
                    "retained_kv": retained,
                    "estimated_kv_bytes": estimate_kv_bytes(
                        retained, model.cfg.n_layers, model.cfg.n_heads, model.head_dim, args.bytes_per_value
                    ),
                }
            )

    acc_df = pd.DataFrame(rows)
    mem_df = pd.DataFrame(memory_rows)
    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "checkpoint_train_args": train_args,
        "task": args.task,
        "background": args.background,
        "seq_lengths": args.seq_lengths,
        "budgets": args.budgets,
        "policies": args.policies,
        "num_examples": args.num_examples,
        "note": "simulation masks the final query attention to retained positions; incremental uses actual pruned past_key_values",
        "best_by_seq_len": acc_df.sort_values("accuracy", ascending=False).groupby("seq_len").head(1).to_dict(orient="records"),
    }
    return acc_df, mem_df, summary


@torch.no_grad()
def predict_plain(model, prompt_ids, device):
    prompt = prompt_ids[None, :].to(device)
    out = model(prompt)
    logits = out["logits"][:, -1, :]
    return int(logits.argmax(dim=-1).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--mode", choices=["simulation", "incremental"], default="simulation")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bytes-per-value", type=int, default=4)
    parser.add_argument("--debug-plain", action="store_true")
    parser.add_argument("--background", choices=["random", "fasta"], default="random")
    parser.add_argument("--fasta-path")
    parser.add_argument("--task", choices=TASKS, default="motif_retrieval")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if args.policies is None:
        args.policies = list(DEFAULT_POLICIES)
    if args.include_oracle and "attention_oracle" not in args.policies:
        args.policies.append("attention_oracle")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    acc_df, mem_df, summary = evaluate(args)
    acc_df.to_csv(results_dir / "accuracy_vs_budget.csv", index=False)
    mem_df.to_csv(results_dir / "memory_vs_length.csv", index=False)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(acc_df[["mode", "seq_len", "kv_budget", "policy", "accuracy", "avg_retained_kv", "compression_ratio_vs_full"]].to_string(index=False))
    print(f"wrote {results_dir / 'accuracy_vs_budget.csv'}, {results_dir / 'memory_vs_length.csv'}, {results_dir / 'summary.json'}")

    try:
        from plots import make_plots

        make_plots(str(results_dir))
    except Exception as exc:  # pragma: no cover - plotting should not block eval.
        print(f"plot generation skipped: {exc}")




if __name__ == "__main__":
    main()
