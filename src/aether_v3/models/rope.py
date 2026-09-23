"""Rotary position embeddings (GPT-NeoX / Llama / Qwen "rotate_half" convention).

Using the same convention Qwen3 uses (rather than the original interleaved
RoPE layout) so `AetherSpeech`'s positional handling matches the LM it will
eventually feed into (phase 2), even though phase 1 only trains the CTC
branch.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns RoPE values for ``[position_offset, position_offset + seq_len)``."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(
        position_offset, position_offset + seq_len, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim / 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., seq_len, head_dim); cos/sin: (seq_len, head_dim), broadcast over leading dims."""
    return x * cos + rotate_half(x) * sin
