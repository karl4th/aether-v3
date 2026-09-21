"""AetherSpeech: bidirectional transformer encoder over semantic embeddings.

8 blocks, dim 768, 12 heads, FFN 3072, pre-LN, GELU, RoPE (no causal mask —
this encoder is bidirectional, unlike the downstream autoregressive LM).
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether_v3.config import AetherSpeechConfig
from aether_v3.models.rope import apply_rope, build_rope_cache


class TransformerBlockConfig(Protocol):
    """Structural type for what `TransformerBlock` reads off its `cfg`
    argument - satisfied by `AetherSpeechConfig` as well as the smaller
    private config dataclasses `CTCUpsampler`/`AetherResampler` build their
    inner blocks with (they don't need the rest of `AetherSpeechConfig`'s
    fields, e.g. `semantic_vocab_size`)."""

    hidden_size: int
    num_heads: int
    ffn_size: int
    dropout: float


class RopeSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_bias = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) bool, True = valid token.
            attn_bias = torch.zeros(b, 1, 1, t, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = RopeSelfAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.ffn = FeedForward(cfg.hidden_size, cfg.ffn_size, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AetherSpeechEncoder(nn.Module):
    def __init__(self, cfg: AetherSpeechConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.semantic_embedding = nn.Embedding(cfg.semantic_vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.head_dim = cfg.hidden_size // cfg.num_heads

    def forward(
        self, semantic_codes: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, t = semantic_codes.shape
        x = self.semantic_embedding(semantic_codes)
        cos, sin = build_rope_cache(t, self.head_dim, self.cfg.rope_theta, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
        return self.final_norm(x)
