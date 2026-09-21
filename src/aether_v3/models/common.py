"""Small building blocks shared by Stage 2 modules (AetherResampler, AetherBridge).

Not used by Stage 1 (AetherSpeechEncoder/CTCUpsampler use LayerNorm + GELU,
matching the original phase-1 spec) - RMSNorm + SwiGLU are what the Stage 2
spec (docs/stage2_spec.md Sec.10, 12) calls for in the new Resampler/Bridge
modules specifically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLUMLP(nn.Module):
    """`out_proj(silu(gate) * up)` where `gate, up = in_proj(x).chunk(2)`."""

    def __init__(self, in_dim: int, intermediate_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, 2 * intermediate_dim)
        self.out_proj = nn.Linear(intermediate_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * up)
