"""Greedy CTC decoding: argmax -> collapse repeats -> drop blank -> UTF-8 bytes."""

from __future__ import annotations

import torch

from aether_v3.data.tokenizer import BLANK_ID, byte_ids_to_text


def greedy_ctc_decode(
    log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_id: int = BLANK_ID
) -> list[str]:
    """log_probs: (batch, time, vocab) on CPU. Returns one decoded string per example.

    Predictions are collapsed byte sequences that may not be valid UTF-8
    (especially early in training) - decoding uses `errors="replace"`.
    """
    pred_ids = log_probs.argmax(dim=-1)  # (batch, time)
    texts = []
    for i in range(pred_ids.shape[0]):
        length = int(input_lengths[i])
        seq = pred_ids[i, :length].tolist()
        collapsed: list[int] = []
        prev = None
        for tok in seq:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        texts.append(byte_ids_to_text(collapsed))
    return texts
