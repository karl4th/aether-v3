"""Collate function for `Stage2CachedDataset` batches.

Only pads speech states and target ids and builds their attention masks -
does *not* assemble Qwen's `inputs_embeds` (task prefix + boundary
embeddings + speech + target). That assembly is `AetherSpeechLLM
.build_inputs_embeds`'s job (it already needs the LLM's own embedding
table for the prefix/target text, so doing it here too would just
duplicate that logic) - see docs/stage2_spec.md Sec.3.3. The shared task
prefix is also not this function's concern: it's constant for an entire
run, so the training loop tokenizes it once and passes `prefix_ids`/
`prefix_mask` into the model alongside this collator's output, rather than
recomputing/re-padding it per batch.
"""

from __future__ import annotations

from typing import Any

import torch


def collate_stage2_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    speech_lengths = torch.tensor([r["speech_length"] for r in batch], dtype=torch.long)
    target_lengths = torch.tensor([r["target_ids"].numel() for r in batch], dtype=torch.long)

    hidden_size = batch[0]["speech_states"].shape[-1]
    max_speech_len = int(speech_lengths.max())
    max_target_len = int(target_lengths.max())

    speech_states = torch.zeros(
        len(batch), max_speech_len, hidden_size, dtype=batch[0]["speech_states"].dtype
    )
    speech_mask = torch.zeros(len(batch), max_speech_len, dtype=torch.bool)
    target_ids = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_target_len, dtype=torch.bool)
    has_prefix = "prefix_ids" in batch[0]
    if has_prefix:
        prefix_lengths = torch.tensor([r["prefix_ids"].numel() for r in batch], dtype=torch.long)
        max_prefix_len = int(prefix_lengths.max())
        prefix_ids = torch.zeros(len(batch), max_prefix_len, dtype=torch.long)
        prefix_mask = torch.zeros(len(batch), max_prefix_len, dtype=torch.bool)

    for i, record in enumerate(batch):
        s_len = int(record["speech_length"])
        t_len = int(record["target_ids"].numel())
        speech_states[i, :s_len] = record["speech_states"]
        speech_mask[i, :s_len] = True
        target_ids[i, :t_len] = record["target_ids"]
        target_mask[i, :t_len] = True
        if has_prefix:
            p_len = int(record["prefix_ids"].numel())
            prefix_ids[i, :p_len] = record["prefix_ids"]
            prefix_mask[i, :p_len] = True

    result = {
        "speech_states": speech_states,
        "speech_mask": speech_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "sample_ids": [r["sample_id"] for r in batch],
    }
    if has_prefix:
        result["prefix_ids"] = prefix_ids
        result["prefix_mask"] = prefix_mask
        result["references"] = [r.get("references", [r.get("transcript", "")]) for r in batch]
    return result
