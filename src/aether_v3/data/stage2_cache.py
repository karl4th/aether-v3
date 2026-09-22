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

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from datasets import load_from_disk

from aether_v3.data.tokenizer import byte_ids_to_text
from aether_v3.training.stage2_utils import answer_strings, first_answer, qa_prefix

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


def build_slue_sqa5_shards(
    rows: Any,
    mimi: Any,
    encoder: nn.Module,
    tokenizer: Stage2Tokenizer,
    output_dir: str | Path,
    role: str,
    device: str = "cuda",
    shard_size: int = 256,
    encode_batch_size: int = 16,
    max_examples: int | None = None,
) -> int:
    """Build restartable Stage 2B shards from streamed SLUE-SQA-5 rows.

    Only question audio is encoded. The linked document remains text and is
    tokenized into the per-example prefix. Existing complete shards are left
    untouched, so a Colab preparation run can resume safely.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = encoder.to(device).eval()
    records: list[dict[str, Any]] = []
    existing_paths = sorted(output_dir.glob("shard-*.pt"))
    written = sum(len(load_stage2_cache(path)) for path in existing_paths)
    rows_to_skip = written
    pending_rows: list[dict[str, Any]] = []
    pending_waveforms: list[np.ndarray] = []

    def flush() -> None:
        nonlocal records, written
        if not records:
            return
        shard_index = written // shard_size
        path = output_dir / f"shard-{shard_index:06d}.pt"
        if not path.exists():
            torch.save(records, path)
        written += len(records)
        records = []

    def encode_with_oom_retry(waveforms: list[np.ndarray]) -> list[np.ndarray]:
        try:
            return mimi.encode_semantic(waveforms)
        except torch.cuda.OutOfMemoryError:
            if len(waveforms) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(waveforms) // 2
            return encode_with_oom_retry(waveforms[:middle]) + encode_with_oom_retry(
                waveforms[middle:]
            )

    def process_pending() -> None:
        nonlocal pending_rows, pending_waveforms
        if not pending_rows:
            return
        codes_list = encode_with_oom_retry(pending_waveforms)
        lengths = [len(codes) for codes in codes_list]
        max_length = max(lengths)
        code_tensor = torch.zeros(len(codes_list), max_length, dtype=torch.long, device=device)
        mask = torch.zeros_like(code_tensor, dtype=torch.bool)
        for index, codes in enumerate(codes_list):
            length = lengths[index]
            code_tensor[index, :length] = torch.as_tensor(codes, dtype=torch.long, device=device)
            mask[index, :length] = True
        with torch.no_grad():
            batch_states = encoder(code_tensor, mask).to("cpu", dtype=torch.float16)

        for index, row in enumerate(pending_rows):
            answers = answer_strings(row["answer_spans"])
            answer = first_answer(row["answer_spans"])
            prefix = qa_prefix(row["raw_document_text"])
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            length = lengths[index]
            records.append(
                {
                    "sample_id": row["question_id"],
                    "speech_states": batch_states[index, :length].clone(),
                    "speech_length": length,
                    "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
                    "target_ids": torch.tensor(target_ids, dtype=torch.long),
                    "references": answers,
                }
            )
            if len(records) >= shard_size:
                flush()
        pending_rows = []
        pending_waveforms = []

    for row_index, row in enumerate(rows):
        if row_index < rows_to_skip:
            continue
        if max_examples is not None and written + len(records) + len(pending_rows) >= max_examples:
            break
        audio = row["question_audio"]
        waveform = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
        if sample_rate != mimi.target_sample_rate:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sample_rate, mimi.target_sample_rate
            ).numpy()
        # Validate targets before spending GPU time on this row.
        first_answer(row["answer_spans"])
        pending_rows.append(row)
        pending_waveforms.append(waveform)
        if len(pending_rows) >= encode_batch_size:
            process_pending()
    process_pending()
    flush()
    return written
