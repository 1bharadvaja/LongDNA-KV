"""Small FASTA utilities for real genomic background windows.

Biopython is intentionally not required. The parser here is enough for local
`.fa` and `.fasta` files used as background sequence sources in this demo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


VALID_BASES = {"A", "C", "G", "T", "N"}


@dataclass(frozen=True)
class FastaRecord:
    source_name: str
    record_id: str
    sequence: str


def normalize_dna(seq: str) -> str:
    return "".join(ch if ch in VALID_BASES else "N" for ch in seq.upper())


def gc_content(seq: str) -> float:
    bases = [ch for ch in seq.upper() if ch in {"A", "C", "G", "T"}]
    if not bases:
        return 0.0
    return sum(ch in {"G", "C"} for ch in bases) / len(bases)


def parse_fasta(path: str | Path) -> List[FastaRecord]:
    path = Path(path)
    records: List[FastaRecord] = []
    current_id: str | None = None
    chunks: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(
                        FastaRecord(
                            source_name=path.name,
                            record_id=current_id,
                            sequence=normalize_dna("".join(chunks)),
                        )
                    )
                current_id = line[1:].split()[0] or f"record_{len(records)}"
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        records.append(FastaRecord(source_name=path.name, record_id=current_id, sequence=normalize_dna("".join(chunks))))

    records = [record for record in records if record.sequence]
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def sample_fasta_window(
    records: Iterable[FastaRecord],
    seq_len: int,
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    rng = rng or random
    candidates = [record for record in records if len(record.sequence) >= seq_len]
    if not candidates:
        longest = max((len(record.sequence) for record in records), default=0)
        raise ValueError(f"No FASTA record has at least seq_len={seq_len} bases; longest record has {longest}")

    record = rng.choice(candidates)
    start = rng.randint(0, len(record.sequence) - seq_len)
    end = start + seq_len
    window = record.sequence[start:end]
    metadata = {
        "background_source": "fasta",
        "source_name": record.source_name,
        "record_id": record.record_id,
        "window_start": start,
        "window_end": end,
        "gc_content": gc_content(window),
    }
    return window, metadata
