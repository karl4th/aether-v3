"""Collate function for `CTCCachedDataset` batches.

Semantic code id 0 is a real vocabulary entry (not "empty"), so padding is
tracked with an explicit boolean attention mask rather than a sentinel pad
value. Targets are returned CTC-ready: concatenated into one 1D tensor plus
per-example lengths, matching `torch.nn.functional.ctc_loss`'s expected
input shape.
"""
from __future__ import annotations

import torch


def collate_ctc_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    codes_list, target_list = zip(*batch)

    input_lengths = torch.tensor([c.numel() for c in codes_list], dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in target_list], dtype=torch.long)

    max_input_len = int(input_lengths.max())
    padded_codes = torch.zeros(len(codes_list), max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(len(codes_list), max_input_len, dtype=torch.bool)
    for i, codes in enumerate(codes_list):
        padded_codes[i, : codes.numel()] = codes
        attention_mask[i, : codes.numel()] = True

    targets = torch.cat(target_list) if target_list else torch.zeros(0, dtype=torch.long)

    return {
        "semantic_codes": padded_codes,
        "attention_mask": attention_mask,
        "input_lengths": input_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
    }
