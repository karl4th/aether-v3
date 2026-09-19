"""CTC head: Linear(hidden_size, vocab_size) + a standalone loss function.

The loss is a free function (not a method that would need to reach into a
DDP-wrapped model's inner module) so training code always computes log-probs
via `model(...)` — going through `DistributedDataParallel`'s `__call__` so
its gradient-sync hooks actually fire — and then calls this function with
plain tensors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden_states)
        return F.log_softmax(logits, dim=-1)


def compute_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """log_probs: (batch, time, vocab). Internally transposed to (time, batch, vocab)."""
    log_probs_tbv = log_probs.transpose(0, 1)
    return F.ctc_loss(
        log_probs_tbv,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )
