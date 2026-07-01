"""Train a tiny DNA Transformer on controlled long-range DNA probes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import trange

from data import STOI, TASKS, VOCAB, generate_batch
from model import ModelConfig, TinyDNATransformer
from real_data import parse_fasta


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def accuracy(
    model: TinyDNATransformer,
    seq_len: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    *,
    task: str,
    background: str,
    fasta_records: list | None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    rng = random.Random(12345)
    for _ in range(batches):
        x, y = generate_batch(
            batch_size,
            seq_len,
            device=device,
            rng=rng,
            task=task,
            background=background,
            fasta_records=fasta_records,
        )
        logits = model(x)["logits"][:, -1, :]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    model.train()
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--out", default="results/model.pt")
    parser.add_argument("--background", choices=["random", "fasta"], default="random")
    parser.add_argument("--fasta-path")
    parser.add_argument("--task", choices=TASKS, default="motif_retrieval")
    args = parser.parse_args()

    device = pick_device(args.device)
    Path("results").mkdir(exist_ok=True)
    torch.manual_seed(7)
    random.seed(7)
    fasta_records = None
    if args.background == "fasta":
        if not args.fasta_path:
            raise ValueError("--background fasta requires --fasta-path")
        fasta_records = parse_fasta(args.fasta_path)
        print(f"Loaded {len(fasta_records)} FASTA record(s) from {args.fasta_path}")
    else:
        print("Warning: using toy random DNA background. Use --background fasta --fasta-path ... for real sequence backgrounds.")
    print(f"Training task={args.task} background={args.background} seq_len={args.seq_len} device={device}")

    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=4 * args.d_model,
        max_seq_len=args.max_seq_len,
    )
    model = TinyDNATransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(7)

    pbar = trange(1, args.steps + 1, desc="train")
    for step in pbar:
        x, y = generate_batch(
            args.batch_size,
            args.seq_len,
            device=device,
            rng=rng,
            task=args.task,
            background=args.background,
            fasta_records=fasta_records,
        )
        out = model(x, targets=y)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_every == 0:
            train_acc = accuracy(
                model,
                args.seq_len,
                args.batch_size,
                batches=2,
                device=device,
                task=args.task,
                background=args.background,
                fasta_records=fasta_records,
            )
            val_acc = accuracy(
                model,
                args.seq_len,
                args.batch_size,
                batches=4,
                device=device,
                task=args.task,
                background=args.background,
                fasta_records=fasta_records,
            )
            pbar.set_postfix(loss=f"{loss.item():.3f}", train_acc=f"{train_acc:.2%}", val_acc=f"{val_acc:.2%}")

    checkpoint = {
        "model_config": cfg.to_dict(),
        "state_dict": model.state_dict(),
        "vocab": STOI,
        "train_args": vars(args),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    config_path = out_path.with_suffix(".train_config.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({"device": str(device), **vars(args)}, f, indent=2)
    print(f"saved checkpoint to {out_path}")
    print(f"saved train config to {config_path}")


if __name__ == "__main__":
    main()
