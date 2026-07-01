# LongDNA-KV

LongDNA-KV is a small PyTorch research prototype for biology-aware KV-cache retention in long-context DNA inference. It trains a tiny causal Transformer on controlled long-range probes embedded in either random DNA or local FASTA genomic backgrounds, then compares which old key/value entries are retained under the same fixed memory budget.

## Motivation

Full KV caching remembers every token but grows linearly with context length. A sliding window saves memory, but it can evict old genomic evidence needed by a final query. LongDNA-KV isolates this retention problem: the target is synthetic and known, while the background can expose policies to more realistic k-mer, repeat, and GC statistics.

This is not a promoter classifier, enhancer benchmark, or SOTA biology claim. It is a controlled KV-retention probe embedded in DNA-like sequence.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Random background, original motif retrieval task:

```bash
python train.py --seq-len 512 --task motif_retrieval --background random
```

FASTA background with distractors:

```bash
python train.py --seq-len 1024 --task distractor_motif_retrieval --background fasta --fasta-path data/example.fa --out results/model_fasta_1024.pt
```

Training prints the sequence length, task, background, and a warning when using toy random DNA.

## Evaluate

Keep evaluation at the trained sequence length unless you intentionally want an out-of-distribution test:

```bash
python eval.py --seq-lengths 512 --budgets 64 128 256 --num-examples 100
```

For a FASTA-trained checkpoint:

```bash
python eval.py --checkpoint results/model_fasta_1024.pt --seq-lengths 1024 --budgets 64 128 256 --num-examples 100 --task distractor_motif_retrieval --background fasta --fasta-path data/example.fa
```

`eval.py` prints the checkpoint training `seq_len` and warns when an eval length differs. Defaults are intentionally small and exclude `attention_oracle`, because the oracle runs full attention with `output_attentions=True` and can be memory-heavy. Add it explicitly:

```bash
python eval.py --seq-lengths 512 --budgets 128 --num-examples 20 --include-oracle
```

Outputs:

- `results/accuracy_vs_budget.csv`
- `results/memory_vs_length.csv`
- `results/summary.json`
- `results/accuracy_vs_budget_len*.png`
- `results/memory_vs_length.png`

## Streamlit Demo

```bash
streamlit run app.py
```

The app shows the probe task, background type, FASTA record/window metadata when available, motif and distractor positions, the model prediction, estimated KV memory, compression ratio, and a retained-token strip.

## Probe Tasks

- `motif_retrieval`: an early motif determines the held-out final target token.
- `distractor_motif_retrieval`: the true early motif determines the target, while later distractor motifs try to confuse retention.
- `reverse_complement_retrieval`: motif families include reverse-complement variants.
- `paired_motif_dependency`: the target depends on one early motif and one middle motif.
- `old_region_gc_bucket`: the target depends on GC content in an old region near the beginning.

The final `NNNNNNNNNNNN` region is a query region in the prompt, not padding. The held-out target is predicted with `logits[:, -1, :]`.

## KV Policies

- `full_cache`: retains all previous K/V entries.
- `sliding_window`: retains only the most recent `B` positions.
- `recent_plus_uniform_stride`: retains a dense recent window plus evenly spaced older positions.
- `recent_plus_rare_kmer`: retains a dense recent window plus older positions whose local k-mer is rare in the sequence.
- `recent_plus_motif`: retains a dense recent window plus older positions near known synthetic motif hits, including reverse-complement hits.
- `attention_oracle`: analysis-only upper bound. It runs full-cache attention once and keeps top-attended old positions under the same budget.

All non-full policies respect the same total KV budget.

## Expected Shape

After training and evaluating at the same `seq_len`, the desired proof-of-concept shape is:

| policy | expected behavior |
| --- | --- |
| `full_cache` | high accuracy, highest memory |
| `sliding_window` | low accuracy when the early evidence is evicted |
| `recent_plus_uniform_stride` | sometimes recovers part of the signal |
| `recent_plus_rare_kmer` | often preserves useful old DNA positions |
| `recent_plus_motif` | strong on motif-based probes with much lower memory than full cache |

## Limitations

- The long-range dependency is synthetic even when the background is real FASTA.
- The motif-aware policy knows the synthetic motifs, so it is a controlled retention demonstration, not motif discovery.
- `attention_oracle` is not deployable because it needs a full-cache pass first.
- Incremental mode uses actual pruned `past_key_values`, but long token-by-token evaluation is slow in Python.
- The bundled `data/example.fa` is a small toy FASTA so the repo runs offline; use your own FASTA for more realistic backgrounds.
