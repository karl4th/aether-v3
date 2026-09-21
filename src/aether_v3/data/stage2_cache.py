"""Stage 2 speech-state cache builder.

Deliberately does *not* touch raw audio or Mimi at all: it consumes the
already-extracted Stage 1 cache (`semantic_codes`/`byte_target`, see
`mimi_cache.py`) directly, so Stage 2 samples go through exactly the same
audio decode/resample/Mimi-encode path Stage 1 used - no new normalization
introduced anywhere in the chain (docs/stage2_spec.md Sec.11 "cache
builder должен взять точно тот же input preprocessing"). Only runs the
(frozen) `AetherSpeechEncoder` on top of those existing semantic codes.

Note: the current (post-CTC-upsampler-fix) `train` cache keeps ~99.99% of
examples (see docs/stage1_report.md Sec.5) - safe to reuse for Stage 2
training as-is; the tiny remainder dropped there was for CTC-feasibility
reasons irrelevant to Stage 2's LM loss anyway.

Output is a plain list of dicts (`Stage2CacheRecord`-shaped), saved with
`torch.save` - not an Arrow dataset. This is deliberately the simplest
thing that works for Phase 0-3 scale (64-128 samples for tiny-overfit, a
few thousand for 2A/2B probes - see docs/stage2_spec.md Sec.6.1); sharding
or streaming for the full ~460h run is explicitly out of scope for v0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from datasets import load_from_disk

from aether_v3.data.tokenizer import byte_ids_to_text

logger = logging.getLogger(__name__)


class Stage2Tokenizer(Protocol):
    """The minimal interface `build_stage2_cache` needs from a tokenizer -
    satisfied by any real Hugging Face tokenizer, and by a trivial fake in
    tests (no need to download real Qwen tokenizer files for a unit test)."""

    eos_token_id: int

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]: ...


def build_stage2_cache(
    stage1_cache_role_path: str | Path,
    encoder: nn.Module,
    tokenizer: Stage2Tokenizer,
    role: str,
    device: str = "cpu",
    batch_size: int = 16,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Runs the frozen `AetherSpeechEncoder` over a Stage 1 cache role and
    recovers the human-readable transcript + Qwen target token ids.

    `encoder` must already be the trained/frozen `AetherSpeechEncoder` (not
    wrapped in `AetherCTCModel` - just the encoder). Text is recovered via
    `byte_ids_to_text(byte_target)`, an exact inverse of the
    `text_to_byte_ids` call `mimi_cache.py` used to build `byte_target` in
    the first place - lossless, since it's literally the same UTF-8 bytes.
    """
    raw = load_from_disk(str(stage1_cache_role_path))
    raw.set_format("python")
    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))

    encoder = encoder.to(device)
    encoder.eval()

    records: list[dict[str, Any]] = []
    n_total = len(raw)
    for start in range(0, n_total, batch_size):
        rows = raw[start : start + batch_size]
        codes_list: list[list[int]] = rows["semantic_codes"]
        byte_targets: list[list[int]] = rows["byte_target"]

        lengths = [len(c) for c in codes_list]
        max_len = max(lengths)
        codes = torch.zeros(len(codes_list), max_len, dtype=torch.long)
        mask = torch.zeros(len(codes_list), max_len, dtype=torch.bool)
        for i, c in enumerate(codes_list):
            codes[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            mask[i, : len(c)] = True
        codes = codes.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            hidden = encoder(codes, mask)  # (B, T, H)
        hidden = hidden.detach().to("cpu", dtype=torch.float16)

        for i, length in enumerate(lengths):
            transcript = byte_ids_to_text(byte_targets[i])
            target_ids = tokenizer(transcript, add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            sample_index = start + i
            records.append(
                {
                    "sample_id": f"{role}_{sample_index:06d}",
                    "speech_states": hidden[i, :length].clone(),
                    "speech_length": length,
                    "transcript": transcript,
                    "target_ids": torch.tensor(target_ids, dtype=torch.long),
                }
            )

        if (start // batch_size) % 20 == 0:
            logger.info(
                "%s: encoded %d/%d examples", role, min(start + batch_size, n_total), n_total
            )

    return records


def save_stage2_cache(records: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, str(path))


def load_stage2_cache(path: str | Path) -> list[dict[str, Any]]:
    return torch.load(str(path), weights_only=False)
