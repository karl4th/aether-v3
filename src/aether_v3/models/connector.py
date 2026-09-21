"""AetherConnector: AetherResampler -> AetherBridge + speech boundary embeddings.

Stage 2 spec Sec.6, 14. The boundary embeddings (`<SPEECH_START>`/
`<SPEECH_END>`) live here rather than in `AetherSpeechLLM` because they are
in the LLM's embedding space (`bridge.output_dim`), same as the projected
speech states - they bracket the speech segment inside the LLM's
`inputs_embeds` sequence, not inside the vocabulary (Sec.14: "не обязательно
новые vocabulary tokens Qwen").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_v3.config import ConnectorConfig
from aether_v3.models.bridge import AetherBridge
from aether_v3.models.resampler import AetherResampler


class AetherConnector(nn.Module):
    def __init__(self, speech_hidden_size: int, cfg: ConnectorConfig) -> None:
        super().__init__()
        self.resampler = AetherResampler(speech_hidden_size, cfg.resampler)
        self.bridge = AetherBridge(cfg.bridge)
        self.speech_start = nn.Parameter(torch.zeros(cfg.bridge.output_dim))
        self.speech_end = nn.Parameter(torch.zeros(cfg.bridge.output_dim))

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns `(speech_embeds, speech_mask)`, both in the LLM's
        embedding space / at the (possibly resampled) speech rate -
        boundary embeddings are not inserted here (see `AetherSpeechLLM
        .build_inputs_embeds`, which needs per-example lengths anyway to
        assemble the full prefix/speech/target sequence)."""
        x, mask = self.resampler(hidden_states, attention_mask)
        return self.bridge(x), mask
