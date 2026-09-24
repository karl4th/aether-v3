"""CTC-only temporal upsampler.

Expands `AetherSpeechEncoder`'s 12.5Hz states to `upsample_factor`x the
rate (e.g. 50Hz at factor=4) before the CTC head, without touching
`AetherSpeechEncoder`'s own output - the future AetherBridge/Qwen stage
(phase 2) still consumes that at the original 12.5Hz.

Why this exists: CTC needs at least one input frame per target label
(plus separators for adjacent repeats - see `_ctc_min_input_length` in
`aether_v3.data.mimi_cache`), and Mimi's 12.5 frames/sec is below typical
English byte-rate (~12-18 bytes/sec) - a large fraction of real
utterances are structurally CTC-infeasible at the raw frame rate,
regardless of model capacity. Measured on the actual train cache: only
~15.7% of examples survived the (pre-upsampler) CTC-feasibility filter,
and even those sat at a mean required/available frame ratio of ~0.89
(p95/p99 at the 1.0 ceiling) - essentially no slack for blank insertions.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from aether_v3.config import CTCConfig
from aether_v3.models.aether_speech import TransformerBlock
from aether_v3.models.rope import build_rope_cache


@dataclasses.dataclass
class _UpsamplerBlockConfig:
    """Just the fields `TransformerBlock` reads off its `cfg` argument."""

    hidden_size: int
    num_heads: int
    ffn_size: int
    dropout: float


class CTCUpsampler(nn.Module):
    def __init__(self, hidden_size: int, cfg: CTCConfig) -> None:
        super().__init__()
        if cfg.lookahead_frames < 0:
            raise ValueError("lookahead_frames must be non-negative")
        self.upsample_factor = cfg.upsample_factor
        self.lookahead_frames = cfg.lookahead_frames
        self.proj = nn.Linear(hidden_size, hidden_size * cfg.upsample_factor)
        # One learned embedding per subframe slot (shared across all
        # timesteps) so the four positions produced from a single 80ms
        # AetherSpeech state can specialize instead of starting identical.
        self.subframe_pos_emb = nn.Parameter(torch.zeros(cfg.upsample_factor, hidden_size))
        block_cfg = _UpsamplerBlockConfig(
            hidden_size=hidden_size,
            num_heads=cfg.upsampler_num_heads,
            ffn_size=cfg.upsampler_ffn_size,
            dropout=cfg.upsampler_dropout,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(cfg.upsampler_num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.rope_theta = cfg.upsampler_rope_theta
        self.head_dim = hidden_size // cfg.upsampler_num_heads

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        b, t, h = hidden_states.shape
        u = self.upsample_factor

        x = self.proj(hidden_states).view(b, t, u, h)
        x = x + self.subframe_pos_emb.view(1, 1, u, h)
        # (B, T, U, H) -> (B, T*U, H); t-major order preserves the
        # right-padded-tail invariant collate_ctc_batch relies on (each
        # original frame's U subframes stay contiguous and in order, so a
        # valid prefix of length input_lengths[i] at 12.5Hz is still a
        # valid prefix of length input_lengths[i]*U after upsampling).
        x = x.reshape(b, t * u, h)

        up_mask = None
        if attention_mask is not None:
            up_mask = attention_mask.unsqueeze(-1).expand(b, t, u).reshape(b, t * u)

        cos, sin = build_rope_cache(t * u, self.head_dim, self.rope_theta, x.device, x.dtype)
        # Convert bounded Mimi-frame lookahead to the upsampled time axis.
        # This remains streamable by delaying emission until those frames
        # arrive; unlike bidirectional attention it never sees the rest of
        # the utterance.
        right_context = self.lookahead_frames * u
        for block in self.blocks:
            x, _ = block(x, cos, sin, up_mask, causal=True, right_context=right_context)
        return self.final_norm(x)
