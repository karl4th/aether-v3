"""Auxiliary prediction of future Mimi semantic tokens.

CTC rewards only transcript-relevant information and maps non-text events to
blank. These small training-only heads ask causal AetherSpeech states to retain
predictive structure from the Mimi q0 stream itself. Multiple horizons reduce
the chance that the objective becomes only a next-token identity shortcut.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPredictionHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, horizons: list[int]) -> None:
        super().__init__()
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("future_horizons must contain positive integers")
        if len(set(horizons)) != len(horizons):
            raise ValueError("future_horizons must not contain duplicates")
        self.horizons = tuple(sorted(horizons))
        self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for _ in self.horizons])

    def loss(
        self,
        hidden_states: torch.Tensor,
        semantic_codes: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        sequence_length = hidden_states.shape[1]
        for horizon, head in zip(self.horizons, self.heads, strict=True):
            if horizon >= sequence_length:
                continue
            source = hidden_states[:, :-horizon]
            targets = semantic_codes[:, horizon:]
            valid = attention_mask[:, :-horizon] & attention_mask[:, horizon:]
            if not bool(valid.any()):
                continue
            logits = head(source)
            token_losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).view_as(targets)
            losses.append(token_losses[valid].mean())
        if not losses:
            return hidden_states.sum() * 0.0
        return torch.stack(losses).mean()
