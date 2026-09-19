"""AetherCTCModel = semantic embedding + AetherSpeech encoder + CTC head.

This is the whole trainable unit for phase 1. `AetherBridge` and the
Qwen3-4B LM branch from the full architecture are not implemented here —
that's phase 2, gated on this CTC branch reaching decent accuracy on its
own.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from aether_v3.config import AetherSpeechConfig, CTCConfig
from aether_v3.models.aether_speech import AetherSpeechEncoder
from aether_v3.models.ctc_head import CTCHead


class AetherCTCModel(nn.Module):
    def __init__(self, speech_cfg: AetherSpeechConfig, ctc_cfg: CTCConfig) -> None:
        super().__init__()
        self.encoder = AetherSpeechEncoder(speech_cfg)
        self.ctc_head = CTCHead(speech_cfg.hidden_size, ctc_cfg.vocab_size)

    def forward(self, semantic_codes: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(semantic_codes, attention_mask)
        return self.ctc_head(hidden)
