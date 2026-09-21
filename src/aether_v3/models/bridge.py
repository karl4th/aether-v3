"""AetherBridge: projects speech states into the frozen LLM's embedding space.

Stage 2 spec Sec.12-13: `input_dim` (768, AetherSpeech's hidden size) ->
`output_dim` (2560, Qwen3-4B's hidden size) through a single SwiGLU MLP.
Deliberately small (one MLP, no Transformer layers) - if bridging two
representations needs a large standalone network, that is itself a
separate research finding, not something to build speculatively here.

The learned output scale exists because the Bridge's output and Qwen's
text-token embeddings can have very different RMS at initialization; the
scale (and the surrounding RMSNorms) let training close that gap instead
of feeding Qwen's first layer an out-of-distribution input scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_v3.config import BridgeConfig
from aether_v3.models.common import RMSNorm, SwiGLUMLP


class AetherBridge(nn.Module):
    def __init__(self, cfg: BridgeConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.input_dim)
        self.mlp = SwiGLUMLP(cfg.input_dim, cfg.intermediate_dim, cfg.output_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_norm = RMSNorm(cfg.output_dim)
        self.output_scale = nn.Parameter(torch.tensor(float(cfg.init_output_scale)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(hidden_states)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.output_norm(x)
        return x * self.output_scale
