"""KV-cache retention policies for long-context DNA prompts."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from data import MOTIFS, decode, reverse_complement


POLICIES = [
    "full_cache",
    "sliding_window",
    "recent_plus_uniform_stride",
    "recent_plus_rare_kmer",
    "recent_plus_motif",
    "attention_oracle",
]


def estimate_kv_bytes(
    retained_seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bytes_per_value: int = 4,
) -> int:
    return 2 * num_layers * num_heads * retained_seq_len * head_dim * bytes_per_value


def _split_budget(budget: int, recent: int | None) -> tuple[int, int]:
    if budget <= 0:
        return 0, 0
    recent_budget = min(budget, recent if recent is not None else max(1, budget // 2))
    old_budget = max(0, budget - recent_budget)
    return recent_budget, old_budget


def _recent_positions(upto: int, n: int) -> List[int]:
    if n <= 0:
        return []
    return list(range(max(0, upto - n), upto))


def _dedupe_sorted(indices: Iterable[int], upto: int) -> List[int]:
    return sorted({int(i) for i in indices if 0 <= int(i) < upto})


def rare_kmer_scores(tokens: Sequence[int], k: int = 8) -> Dict[int, float]:
    """Score positions by inverse frequency of their local k-mer."""

    seq = decode(tokens)
    if len(seq) < k:
        return {i: 1.0 for i in range(len(seq))}
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    scores: Dict[int, float] = {}
    half = k // 2
    for pos in range(len(seq)):
        start = min(max(0, pos - half), len(seq) - k)
        scores[pos] = 1.0 / counts[seq[start : start + k]]
    return scores


def motif_scores(tokens: Sequence[int], motifs: Sequence[str] | None = None, radius: int = 6) -> Dict[int, float]:
    motifs = motifs or [*MOTIFS, *[reverse_complement(m) for m in MOTIFS]]
    seq = decode(tokens)
    scores: Dict[int, float] = {}
    for motif in motifs:
        start = seq.find(motif)
        while start != -1:
            for pos in range(max(0, start - radius), min(len(seq), start + len(motif) + radius)):
                scores[pos] = max(scores.get(pos, 0.0), 1.0)
            start = seq.find(motif, start + 1)
    return scores


def _top_scored_old(
    scores: Mapping[int, float],
    upto: int,
    old_budget: int,
    exclude: set[int],
) -> List[int]:
    candidates = [pos for pos in range(upto) if pos not in exclude]
    candidates.sort(key=lambda p: (scores.get(p, 0.0), -p), reverse=True)
    return sorted(candidates[:old_budget])


def select_retained_positions(
    tokens: Sequence[int],
    upto: int,
    budget: int,
    policy: str,
    *,
    recent: int | None = None,
    stride: int | None = None,
    kmer_k: int = 8,
    oracle_scores: Mapping[int, float] | None = None,
) -> List[int]:
    """Return sorted past positions from ``[0, upto)`` to keep in the KV cache."""

    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}. Expected one of {POLICIES}")
    upto = max(0, min(int(upto), len(tokens)))

    if policy == "full_cache":
        return list(range(upto))
    if budget <= 0 or upto == 0:
        return []
    if policy == "sliding_window":
        return _recent_positions(upto, min(budget, upto))

    recent_budget, old_budget = _split_budget(budget, recent)
    kept = _recent_positions(upto, recent_budget)
    kept_set = set(kept)
    old_upto = max(0, upto - recent_budget)

    if old_budget <= 0 or old_upto <= 0:
        return _dedupe_sorted(kept, upto)[-budget:]

    if policy == "recent_plus_uniform_stride":
        if stride is None:
            stride = max(1, int(np.ceil(old_upto / old_budget)))
        old = list(range(0, old_upto, stride))[:old_budget]
    elif policy == "recent_plus_rare_kmer":
        old = _top_scored_old(rare_kmer_scores(tokens, k=kmer_k), old_upto, old_budget, kept_set)
    elif policy == "recent_plus_motif":
        old = _top_scored_old(motif_scores(tokens), old_upto, old_budget, kept_set)
    elif policy == "attention_oracle":
        if oracle_scores is None:
            raise ValueError("attention_oracle requires oracle_scores")
        old = _top_scored_old(oracle_scores, old_upto, old_budget, kept_set)
    else:
        old = []

    retained = _dedupe_sorted([*old, *kept], upto)
    if len(retained) > budget:
        retained = retained[-budget:]
    return retained


def retained_strip(seq_len: int, retained: Sequence[int], width: int = 100) -> str:
    """ASCII strip useful in logs and README examples."""

    if seq_len <= 0:
        return ""
    retained_set = set(retained)
    chars = []
    for i in range(width):
        lo = int(i * seq_len / width)
        hi = max(lo + 1, int((i + 1) * seq_len / width))
        chars.append("|" if any(p in retained_set for p in range(lo, hi)) else ".")
    return "".join(chars)
