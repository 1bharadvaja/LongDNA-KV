"""A tiny causal DNA Transformer with reusable and prunable KV caches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ModelConfig:
    vocab_size: int = 5
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 8192

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        return asdict(self)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    head_dim = x.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)[:, None, :, :]
    return (x * emb.cos()) + (_rotate_half(x) * emb.sin())


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        bsz, q_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, position_ids)
        k = apply_rope(k, position_ids)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_len = k.size(2)

        if past_kv is None and q_len > 1:
            causal = torch.tril(torch.ones(q_len, key_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, q_len, d_model)
        y = self.out(y)
        new_kv = (k, v) if use_cache else None
        return y, new_kv, attn if output_attentions else None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        attn_out, new_kv, attn = self.attn(
            self.ln1(x),
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv, attn


class TinyDNATransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def head_dim(self) -> int:
        return self.cfg.d_model // self.cfg.n_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            past_len = 0 if not past_key_values else past_key_values[0][0].size(2)
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)[None, :].expand(bsz, -1)

        x = self.drop(self.token_emb(input_ids))
        new_cache: KVCache = []
        attentions = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, layer_cache, attn = block(
                x,
                position_ids=position_ids,
                past_kv=past,
                use_cache=use_cache,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if use_cache and layer_cache is not None:
                new_cache.append(layer_cache)
            if output_attentions and attn is not None:
                attentions.append(attn)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, -1, :], targets)
        return {"logits": logits, "loss": loss, "past_key_values": new_cache if use_cache else None, "attentions": attentions}


def prune_kv_cache(cache: KVCache, cache_positions: Sequence[int], retained_positions: Sequence[int]) -> KVCache:
    """Gather arbitrary retained positions from each layer's K/V tensors.

    ``cache_positions`` records the absolute sequence position represented by
    every KV slot. Policies operate on those positions; this function converts
    the policy decision back into tensor indices and preserves chronological
    order for the next autoregressive step.
    """

    if not cache:
        return cache
    index_by_pos = {int(pos): i for i, pos in enumerate(cache_positions)}
    gather = [index_by_pos[pos] for pos in retained_positions if pos in index_by_pos]
    if not gather:
        return [(k[:, :, :0, :], v[:, :, :0, :]) for k, v in cache]
    idx = torch.tensor(gather, dtype=torch.long, device=cache[0][0].device)
    return [(k.index_select(2, idx), v.index_select(2, idx)) for k, v in cache]
