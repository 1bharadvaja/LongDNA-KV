"""Synthetic long-range DNA retrieval probes for LongDNA-KV."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


VOCAB: List[str] = ["A", "C", "G", "T", "N"]
STOI: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
ITOS: Dict[int, str] = {i: ch for ch, i in STOI.items()}
DNA_BASES = ["A", "C", "G", "T"]

MOTIF_TO_TARGET: Dict[str, str] = {
    "ACGTACGA": "G",
    "TTGCAACT": "T",
    "GGATCCAA": "A",
    "CCTAGGTT": "C",
}
MOTIFS = list(MOTIF_TO_TARGET)
QUERY = "NNNNNNNNNNNN"
TASKS = [
    "motif_retrieval",
    "distractor_motif_retrieval",
    "reverse_complement_retrieval",
    "paired_motif_dependency",
    "old_region_gc_bucket",
]


@dataclass(frozen=True)
class DNAExample:
    prompt_ids: torch.Tensor
    target_id: int
    sequence: str
    motif: str
    motif_start: int
    query_start: int
    target: str
    background_source: str = "random"
    source_name: str | None = None
    record_id: str | None = None
    window_start: int | None = None
    window_end: int | None = None
    gc_content: float | None = None
    distractor_positions: Tuple[int, ...] = ()
    distractor_motifs: Tuple[str, ...] = ()
    paired_motif: str | None = None
    paired_motif_start: int | None = None
    old_region_start: int | None = None
    old_region_end: int | None = None
    task_name: str = "motif_retrieval"


def encode(seq: str) -> List[int]:
    return [STOI[ch] for ch in seq]


def decode(ids: Sequence[int]) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(table)[::-1]


def _random_background(seq_len: int, rng: random.Random) -> tuple[str, dict]:
    seq = "".join(rng.choice(DNA_BASES) for _ in range(seq_len))
    gc = sum(ch in {"G", "C"} for ch in seq) / len(seq)
    return seq, {"background_source": "random", "gc_content": gc}


def _insert(chars: list[str], pos: int, text: str) -> None:
    chars[pos : pos + len(text)] = list(text)


def _safe_position(rng: random.Random, lo: int, hi: int, width: int) -> int:
    hi = max(lo, hi - width)
    return rng.randint(lo, hi)


def _gc_bucket_target(seq: str) -> str:
    gc = sum(ch in {"G", "C"} for ch in seq) / max(1, sum(ch in {"A", "C", "G", "T"} for ch in seq))
    if gc < 0.40:
        return "A"
    if gc < 0.50:
        return "C"
    if gc < 0.60:
        return "G"
    return "T"


def generate_example(
    seq_len: int = 1024,
    motif: str | None = None,
    motif_start: int | None = None,
    rng: random.Random | None = None,
    *,
    task: str = "motif_retrieval",
    background: str = "random",
    background_sequence: str | None = None,
    background_metadata: dict | None = None,
    num_distractors: int = 3,
) -> DNAExample:
    """Generate one prompt of length ``seq_len`` plus a held-out final target.

    The model sees the prompt and must predict the next token. The query region
    occupies the final prompt tokens, while the answer is determined only by an
    early motif.
    """

    rng = rng or random
    if seq_len < len(QUERY) + 32:
        raise ValueError(f"seq_len must be at least {len(QUERY) + 32}")
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Expected one of {TASKS}")

    if background_sequence is not None:
        if len(background_sequence) < seq_len:
            raise ValueError(f"background_sequence must be at least seq_len={seq_len}")
        base_sequence = background_sequence[:seq_len].upper()
        metadata = dict(background_metadata or {})
        metadata.setdefault("background_source", background)
    else:
        base_sequence, metadata = _random_background(seq_len, rng)

    motif_start = motif_start if motif_start is not None else rng.randint(4, 18)
    query_start = seq_len - len(QUERY)

    chars = [ch if ch in VOCAB else "N" for ch in base_sequence]
    distractor_positions: list[int] = []
    distractor_motifs: list[str] = []
    paired_motif = None
    paired_motif_start = None
    old_region_start = None
    old_region_end = None

    if task == "reverse_complement_retrieval":
        family = motif or rng.choice(MOTIFS)
        inserted_motif = family if rng.random() < 0.5 else reverse_complement(family)
        target = MOTIF_TO_TARGET[family]
        motif = inserted_motif
        _insert(chars, motif_start, motif)
    elif task == "paired_motif_dependency":
        motif = motif or rng.choice(MOTIFS)
        motif_idx = MOTIFS.index(motif)
        paired_motif = rng.choice(MOTIFS)
        paired_idx = MOTIFS.index(paired_motif)
        target = DNA_BASES[(motif_idx + paired_idx) % len(DNA_BASES)]
        paired_motif_start = _safe_position(rng, seq_len // 2, query_start - 16, len(paired_motif))
        _insert(chars, motif_start, motif)
        _insert(chars, paired_motif_start, paired_motif)
    elif task == "old_region_gc_bucket":
        old_region_start = motif_start
        old_region_end = min(query_start, old_region_start + max(24, min(96, seq_len // 8)))
        old_region = "".join(chars[old_region_start:old_region_end])
        target = _gc_bucket_target(old_region)
        motif = f"GC_{target}"
    else:
        motif = motif or rng.choice(MOTIFS)
        target = MOTIF_TO_TARGET[motif]
        _insert(chars, motif_start, motif)

        if task == "distractor_motif_retrieval":
            lo = max(motif_start + len(motif) + 16, seq_len // 4)
            hi = max(lo, query_start - 16)
            for _ in range(num_distractors):
                distractor = rng.choice([m for m in MOTIFS if m != motif])
                pos = _safe_position(rng, lo, hi, len(distractor))
                _insert(chars, pos, distractor)
                distractor_positions.append(pos)
                distractor_motifs.append(distractor)

    _insert(chars, query_start, QUERY)
    sequence = "".join(chars)

    return DNAExample(
        prompt_ids=torch.tensor(encode(sequence), dtype=torch.long),
        target_id=STOI[target],
        sequence=sequence,
        motif=motif,
        motif_start=motif_start,
        query_start=query_start,
        target=target,
        background_source=metadata.get("background_source", background),
        source_name=metadata.get("source_name"),
        record_id=metadata.get("record_id"),
        window_start=metadata.get("window_start"),
        window_end=metadata.get("window_end"),
        gc_content=metadata.get("gc_content"),
        distractor_positions=tuple(distractor_positions),
        distractor_motifs=tuple(distractor_motifs),
        paired_motif=paired_motif,
        paired_motif_start=paired_motif_start,
        old_region_start=old_region_start,
        old_region_end=old_region_end,
        task_name=task,
    )


def generate_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    rng: random.Random | None = None,
    *,
    task: str = "motif_retrieval",
    background: str = "random",
    fasta_records: Sequence[object] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = []
    for _ in range(batch_size):
        if background == "fasta":
            if fasta_records is None:
                raise ValueError("background='fasta' requires fasta_records")
            from real_data import sample_fasta_window

            bg_seq, bg_meta = sample_fasta_window(fasta_records, seq_len, rng=rng)
        else:
            bg_seq, bg_meta = None, None
        examples.append(
            generate_example(
                seq_len=seq_len,
                rng=rng,
                task=task,
                background=background,
                background_sequence=bg_seq,
                background_metadata=bg_meta,
            )
        )
    x = torch.stack([ex.prompt_ids for ex in examples]).to(device)
    y = torch.tensor([ex.target_id for ex in examples], dtype=torch.long, device=device)
    return x, y


def motif_window(example: DNAExample, radius: int = 4) -> range:
    start = max(0, example.motif_start - radius)
    end = min(len(example.sequence), example.motif_start + len(example.motif) + radius)
    return range(start, end)
