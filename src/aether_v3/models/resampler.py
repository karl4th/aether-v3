"""AetherResampler: learned temporal compression of AetherSpeech's 12.5Hz states.

Stage 2 spec Sec.7-11: the R1 vs R4 ablation is the first experiment gate
of Stage 2, so `ratio` (and `enabled`) are the whole point of this module,
not incidental config. Each doubling stage is a strided temporal
convolution (kernel 5, stride 2 - fuses neighbouring frames instead of
average-pooling or dropping them) followed by a small residual
self-attention block for local contextual mixing (phoneme transitions,
word boundaries) that the conv alone can't do. `ratio` is restricted to
powers of two; `ratio=1`/`enabled=False` is an identity pass-through - the
"R1" native-rate experiment arm.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from aether_v3.config import ResamplerConfig
from aether_v3.models.aether_speech import TransformerBlock
from aether_v3.models.common import RMSNorm, SwiGLUMLP
from aether_v3.models.rope import build_rope_cache


@dataclasses.dataclass
class _ResamplerBlockConfig:
    """Just the fields `TransformerBlock` reads off its `cfg` argument."""

    hidden_size: int
    num_heads: int
    ffn_size: int
    dropout: float


class _DownsampleStage(nn.Module):
    """One ×2 stage: strided conv + pointwise SwiGLU + residual local attention."""

    def __init__(
        self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float, conv_kernel: int
    ) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size)
        padding = conv_kernel // 2
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=conv_kernel, stride=2, padding=padding
        )
        self.mlp = SwiGLUMLP(hidden_size, hidden_size * 2, hidden_size)
        block_cfg = _ResamplerBlockConfig(hidden_size, num_heads, ffn_size, dropout)
        self.block = TransformerBlock(block_cfg)
        self.rope_theta = 10000.0
        self.head_dim = hidden_size // num_heads
        self._kernel = conv_kernel
        self._padding = padding

    def _output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """`nn.Conv1d`'s own length formula, applied per-example so the
        post-conv mask reflects each example's real (unpadded) length."""
        return torch.div(lengths + 2 * self._padding - self._kernel, 2, rounding_mode="floor") + 1

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.pre_norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, H) -> (B, T', H)
        x = x + self.mlp(x)

        new_t = x.shape[1]
        new_mask = None
        if mask is not None:
            lengths = mask.sum(dim=1)
            new_lengths = self._output_lengths(lengths).clamp(min=0, max=new_t)
            new_mask = torch.arange(new_t, device=x.device).unsqueeze(0) < new_lengths.unsqueeze(1)

        cos, sin = build_rope_cache(new_t, self.head_dim, self.rope_theta, x.device, x.dtype)
        x = self.block(x, cos, sin, new_mask)
        return x, new_mask


class AetherResampler(nn.Module):
    def __init__(self, hidden_size: int, cfg: ResamplerConfig) -> None:
        super().__init__()
        self.enabled = cfg.enabled and cfg.ratio > 1
        self.ratio = cfg.ratio if self.enabled else 1
        if self.enabled:
            if cfg.ratio & (cfg.ratio - 1) != 0:
                raise ValueError(f"resampler ratio must be a power of two, got {cfg.ratio}")
            num_stages = int(math.log2(cfg.ratio))
            self.stages = nn.ModuleList(
                [
                    _DownsampleStage(
                        hidden_size, cfg.num_heads, cfg.ffn_size, cfg.dropout, cfg.conv_kernel
                    )
                    for _ in range(num_stages)
                ]
            )
        else:
            self.stages = nn.ModuleList()

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.enabled:
            return hidden_states, attention_mask
        x, mask = hidden_states, attention_mask
        for stage in self.stages:
            x, mask = stage(x, mask)
        return x, mask
