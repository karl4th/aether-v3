"""Builds the frozen `stage1_eval_manifest.jsonl` (docs/stage2_spec.md
Sec.6, rule 3 / Sec.12): the exact same eval set (sample IDs, references,
durations) that produced the Stage 1 baseline (WER 16.53% / CER 5.88%),
so no Stage 2 experiment can be compared against a subtly different split
by accident.

The Stage 1 cache (`mimi_cache.py`) only stores `semantic_codes`/
`byte_target` - no sample id or duration - so those have to be recovered
by re-loading the raw LibriSpeech split and re-applying the exact same
duration-bounds filter `_extract_generator` used. Positional alignment
between the recovered raw rows and the cached dataset is *verified*, not
assumed: every recovered reference is re-encoded to UTF-8 bytes and
checked against the cached `byte_target` before being trusted, and any
mismatch aborts loudly rather than silently writing a misaligned
manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from aether_v3.config import DataConfig
from aether_v3.data.librispeech import load_splits
from aether_v3.data.tokenizer import text_to_byte_ids

logger = logging.getLogger(__name__)


def _row_duration_seconds(row: dict[str, Any], expected_sample_rate: int) -> float:
    sr = row["audio"]["sampling_rate"]
    if sr != expected_sample_rate:
        raise ValueError(f"Unexpected sample rate {sr} Hz (expected {expected_sample_rate}).")
    return len(row["audio"]["array"]) / sr


def filter_rows_by_duration(
    rows: list[dict[str, Any]], data_cfg: DataConfig
) -> list[dict[str, Any]]:
    """Re-applies `_extract_generator`'s duration-bounds filter (the only
    filter validation/test cache building applies - see `mimi_cache.py`,
    `drop_ctc_infeasible=False` for those roles) to recover which raw rows
    survived into the cache, in the same order."""
    kept = []
    for row in rows:
        duration = _row_duration_seconds(row, data_cfg.sample_rate_in)
        if data_cfg.min_audio_seconds <= duration <= data_cfg.max_audio_seconds:
            kept.append({"sample_id": row["id"], "reference": row["text"], "duration": duration})
    return kept


def verify_byte_alignment(kept: list[dict[str, Any]], cached_byte_targets: list[list[int]]) -> None:
    """Raises if `kept[i]`'s reference does not re-encode to exactly
    `cached_byte_targets[i]` - the only way to trust that `kept` and the
    cached dataset are positionally aligned."""
    if len(kept) != len(cached_byte_targets):
        raise RuntimeError(
            f"Recovered {len(kept)} raw examples after duration filtering, but the cached "
            f"dataset has {len(cached_byte_targets)} - cannot safely align sample_id/reference "
            "by position. Check that DataConfig matches what built that cache."
        )
    for i, (meta, byte_target) in enumerate(zip(kept, cached_byte_targets, strict=True)):
        if text_to_byte_ids(meta["reference"]) != byte_target:
            raise RuntimeError(
                f"byte_target mismatch at position {i}: the raw split and cached dataset are "
                "not positionally aligned (or the underlying HF dataset changed since the cache "
                "was built)."
            )


def write_manifest(
    kept: list[dict[str, Any]], stage2_cache_path: str | Path, out_path: str | Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, meta in enumerate(kept):
            record = {**meta, "stage2_cache_path": str(stage2_cache_path), "stage2_cache_index": i}
            f.write(json.dumps(record) + "\n")


def build_stage1_eval_manifest(
    data_cfg: DataConfig,
    stage1_cache_role_path: str | Path,
    stage2_cache_path: str | Path,
    out_path: str | Path,
    role: str = "test",
) -> None:
    """Downloads/loads the raw `role` split (network access), recovers
    sample_id/reference/duration, verifies alignment against the existing
    Stage 1 cache at `stage1_cache_role_path`, and writes the manifest."""
    specs = {
        "train": data_cfg.train_splits,
        "validation": data_cfg.validation_splits,
        "test": data_cfg.test_splits,
    }[role]

    raw: Dataset = load_splits(specs, data_cfg.dataset_id, data_cfg.fallback_dataset_id)
    kept = filter_rows_by_duration(list(raw), data_cfg)

    cached = load_from_disk(str(stage1_cache_role_path))
    cached.set_format("python")
    verify_byte_alignment(kept, cached["byte_target"])

    write_manifest(kept, stage2_cache_path, out_path)
    logger.info("Wrote %d-example eval manifest to %s", len(kept), out_path)
