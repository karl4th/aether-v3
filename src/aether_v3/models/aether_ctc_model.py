"""AetherCTCModel = semantic embedding + AetherSpeech encoder + CTC
upsampler + CTC head.

This is the whole trainable unit for phase 1. `AetherBridge` and the
Qwen3-4B LM branch from the full architecture are not implemented here —
that's phase 2, gated on this CTC branch reaching decent accuracy on its
own. Phase 2 will consume `self.encoder`'s raw 12.5Hz output directly;
`self.upsampler` only exists for the CTC branch (see
`aether_v3.models.ctc_upsampler` for why).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_v3.config import AetherSpeechConfig, CTCConfig, SemanticPredictionConfig
from aether_v3.models.aether_speech import AetherSpeechEncoder
from aether_v3.models.ctc_head import CTCHead
from aether_v3.models.ctc_upsampler import CTCUpsampler
from aether_v3.models.semantic_prediction import SemanticPredictionHeads


class AetherCTCModel(nn.Module):
    def __init__(
        self,
        speech_cfg: AetherSpeechConfig,
        ctc_cfg: CTCConfig,
        semantic_cfg: SemanticPredictionConfig | None = None,
    ) -> None:
        super().__init__()
        semantic_cfg = semantic_cfg or SemanticPredictionConfig()
        self.encoder = AetherSpeechEncoder(speech_cfg)
        self.upsampler = CTCUpsampler(speech_cfg.hidden_size, ctc_cfg)
        self.ctc_head = CTCHead(speech_cfg.hidden_size, ctc_cfg.vocab_size)
        self.semantic_prediction = (
            SemanticPredictionHeads(
                speech_cfg.hidden_size,
                speech_cfg.semantic_vocab_size,
                semantic_cfg.future_horizons,
            )
            if semantic_cfg.enabled
            else None
        )
        self.semantic_prediction_weight = semantic_cfg.weight if semantic_cfg.enabled else 0.0
        # Read by train_ctc.py to scale input_lengths (computed at the raw
        # 12.5Hz Mimi rate) up to match this model's upsampled CTC output.
        self.upsample_factor = ctc_cfg.upsample_factor

    def forward(
        self,
        semantic_codes: torch.Tensor,
        attention_mask: torch.Tensor,
        compute_semantic_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        log_probs, hidden = self.forward_with_hidden(semantic_codes, attention_mask)
        if compute_semantic_loss:
            return log_probs, self.semantic_loss(hidden, semantic_codes, attention_mask)
        return log_probs

    def forward_with_hidden(
        self, semantic_codes: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(semantic_codes, attention_mask)
        upsampled = self.upsampler(hidden, attention_mask)
        return self.ctc_head(upsampled), hidden

    def semantic_loss(
        self,
        hidden_states: torch.Tensor,
        semantic_codes: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.semantic_prediction is None:
            return hidden_states.sum() * 0.0
        return self.semantic_prediction.loss(hidden_states, semantic_codes, attention_mask)
