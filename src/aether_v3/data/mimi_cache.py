"""Offline extraction: LibriSpeech audio+text -> cached (semantic_codes, byte_target) Arrow dataset.

Mimi is frozen and would otherwise be re-run on every epoch for no benefit,
so this module runs it once per split and writes the results to disk. This
is meant to be invoked via `scripts/prepare_data.py`, not imported into the
training loop.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import Dataset

from aether_v3.config import DataConfig, ExperimentConfig
from aether_v3.data.librispeech import load_splits
from aether_v3.data.tokenizer import text_to_byte_ids
from aether_v3.models.mimi_wrapper import FrozenMimi

logger = logging.getLogger(__name__)


def _extract_generator(
    raw: Dataset, data_cfg: DataConfig, mimi: FrozenMimi, role: str
) -> Iterator[dict]:
    n_total = len(raw)
    n_dropped = 0
    for start in range(0, n_total, data_cfg.extraction_batch_size):
        end = min(start + data_cfg.extraction_batch_size, n_total)
        chunk = raw[start:end]

        waveforms: list[np.ndarray] = []
        texts: list[str] = []
        for audio, text in zip(chunk["audio"], chunk["text"]):
            wav = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if sr != data_cfg.sample_rate_in:
                raise ValueError(
                    f"Unexpected sample rate {sr} Hz for {role} split "
                    f"(configured sample_rate_in={data_cfg.sample_rate_in})."
                )
            duration = len(wav) / sr
            if not (data_cfg.min_audio_seconds <= duration <= data_cfg.max_audio_seconds):
                n_dropped += 1
                continue
            waveforms.append(wav)
            texts.append(text)

        if not waveforms:
            continue

        code_seqs = mimi.encode_semantic(waveforms, orig_sample_rate=data_cfg.sample_rate_in)
        for codes, text in zip(code_seqs, texts):
            yield {
                "semantic_codes": codes.tolist(),
                "byte_target": text_to_byte_ids(text),
            }

    logger.info(
        "%s: dropped %d/%d examples outside duration bounds [%.1f, %.1f]s",
        role,
        n_dropped,
        n_total,
        data_cfg.min_audio_seconds,
        data_cfg.max_audio_seconds,
    )


def extract_split(specs: list[str], data_cfg: DataConfig, mimi: FrozenMimi, role: str) -> Dataset:
    """Load LibriSpeech split(s) `specs` and extract (semantic_codes, byte_target) pairs."""
    raw = load_splits(specs, data_cfg.dataset_id, data_cfg.fallback_dataset_id)
    return Dataset.from_generator(
        lambda: _extract_generator(raw, data_cfg, mimi, role),
    )


def prepare_cache(config: ExperimentConfig, device: str = "cpu") -> dict[str, Path]:
    """Extract and cache all configured train/validation/test splits.

    Skips any role whose cache directory already exists, so this is safe to
    re-run (e.g. after adding a new split) without redoing finished work.
    """
    mimi = FrozenMimi(config.mimi.pretrained_id, config.mimi.num_quantizers, device=device)
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    role_to_specs = {
        "train": config.data.train_splits,
        "validation": config.data.validation_splits,
        "test": config.data.test_splits,
    }
    out_paths: dict[str, Path] = {}
    for role, specs in role_to_specs.items():
        out_path = cache_dir / role
        out_paths[role] = out_path
        if out_path.exists():
            logger.info("Cache for '%s' already exists at %s, skipping.", role, out_path)
            continue
        logger.info("Extracting '%s' split(s) %s ...", role, specs)
        dataset = extract_split(specs, config.data, mimi, role)
        dataset.save_to_disk(str(out_path))
        logger.info("Saved '%s' cache to %s (%d examples).", role, out_path, len(dataset))
    return out_paths
