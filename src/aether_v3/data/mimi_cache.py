"""Offline extraction: LibriSpeech audio+text -> cached (semantic_codes, byte_target) Arrow dataset.

Mimi is frozen and would otherwise be re-run on every epoch for no benefit,
so this module runs it once per split and writes the results to disk. This
is meant to be invoked via `scripts/prepare_data.py`, not imported into the
training loop.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import Dataset

from aether_v3.config import DataConfig, ExperimentConfig, MimiConfig
from aether_v3.data.librispeech import load_splits
from aether_v3.data.tokenizer import text_to_byte_ids
from aether_v3.models.mimi_wrapper import FrozenMimi

logger = logging.getLogger(__name__)


def _ctc_min_input_length(target_ids: list[int]) -> int:
    """Minimum number of input frames CTC needs to align `target_ids`.

    One frame per label, plus one extra frame between any two adjacent
    identical labels (a blank must separate repeated consecutive labels, or
    they'd collapse into one during decoding).
    """
    if not target_ids:
        return 0
    repeats = sum(1 for a, b in zip(target_ids, target_ids[1:]) if a == b)
    return len(target_ids) + repeats


def _extract_generator(
    raw: Dataset,
    data_cfg: DataConfig,
    mimi_cfg: MimiConfig,
    device: str,
    role: str,
    drop_ctc_infeasible: bool,
) -> Iterator[dict]:
    """Not called directly by `Dataset.from_generator` - see `extract_split`.

    Takes `mimi_cfg`/`device` (plain, trivially-picklable values) and builds
    the actual `FrozenMimi` model *inside* the generator, rather than
    receiving an already-constructed model via a parameter. `datasets`
    hashes every `gen_kwargs` value (via dill) to build its fingerprint, and
    a loaded `MimiModel` can contain `meta` device placeholder tensors (from
    lazy weight init) that dill cannot copy - confirmed empirically, this
    raises `NotImplementedError: Cannot copy out of meta tensor; no data!`
    if a constructed model is passed through instead.
    """
    mimi = FrozenMimi(mimi_cfg.pretrained_id, mimi_cfg.num_quantizers, device=device)
    n_total = len(raw)
    n_dropped_duration = 0
    n_ctc_infeasible = 0
    n_yielded = 0
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
                n_dropped_duration += 1
                continue
            waveforms.append(wav)
            texts.append(text)

        if not waveforms:
            continue

        code_seqs = mimi.encode_semantic(waveforms, orig_sample_rate=data_cfg.sample_rate_in)
        for codes, text in zip(code_seqs, texts):
            byte_target = text_to_byte_ids(text)
            # CTC requires input_length >= target_length (+ separators for
            # adjacent repeated labels); at Mimi's 12.5Hz semantic frame rate,
            # average English text (~13-15 UTF-8 bytes/sec) is close to or
            # above that, a non-trivial fraction of examples can be
            # structurally unalignable. `zero_infinity=True` in the CTC loss
            # would otherwise silently zero these out (no gradient) rather
            # than erroring, so for `train` they're dropped here instead
            # (no point spending a batch slot on zero-gradient examples).
            # `validation`/`test` keep them - WER/CER must reflect real
            # performance on every utterance the model has to transcribe,
            # not just the CTC-alignable ones.
            if _ctc_min_input_length(byte_target) > len(codes):
                n_ctc_infeasible += 1
                if drop_ctc_infeasible:
                    continue
            n_yielded += 1
            yield {
                "semantic_codes": codes.tolist(),
                "byte_target": byte_target,
            }

    logger.info(
        "%s: dropped %d/%d examples outside duration bounds [%.1f, %.1f]s",
        role,
        n_dropped_duration,
        n_total,
        data_cfg.min_audio_seconds,
        data_cfg.max_audio_seconds,
    )
    n_seen = n_yielded + (n_ctc_infeasible if drop_ctc_infeasible else 0)
    if n_seen:
        logger.warning(
            "%s: %d/%d examples (%.1f%%) are CTC-infeasible (target needs more frames than "
            "Mimi provides)%s.",
            role,
            n_ctc_infeasible,
            n_seen,
            100.0 * n_ctc_infeasible / n_seen,
            " - dropped from the cache" if drop_ctc_infeasible else " - kept in the cache",
        )


def extract_split(
    specs: list[str],
    data_cfg: DataConfig,
    mimi_cfg: MimiConfig,
    device: str,
    role: str,
    drop_ctc_infeasible: bool,
) -> Dataset:
    """Load LibriSpeech split(s) `specs` and extract (semantic_codes, byte_target) pairs."""
    raw = load_splits(specs, data_cfg.dataset_id, data_cfg.fallback_dataset_id)
    return Dataset.from_generator(
        _extract_generator,
        gen_kwargs={
            "raw": raw,
            "data_cfg": data_cfg,
            "mimi_cfg": mimi_cfg,
            "device": device,
            "role": role,
            "drop_ctc_infeasible": drop_ctc_infeasible,
        },
    )


def _cache_fingerprint(data_cfg: DataConfig, mimi_cfg: MimiConfig, specs: list[str]) -> str:
    """Hash of every setting that changes what `extract_split` would produce.

    Used to detect a stale cache (e.g. after editing duration bounds or
    splits) instead of silently reusing whatever is already on disk.
    """
    payload = {
        "specs": specs,
        "dataset_id": data_cfg.dataset_id,
        "fallback_dataset_id": data_cfg.fallback_dataset_id,
        "sample_rate_in": data_cfg.sample_rate_in,
        "sample_rate_out": data_cfg.sample_rate_out,
        "min_audio_seconds": data_cfg.min_audio_seconds,
        "max_audio_seconds": data_cfg.max_audio_seconds,
        "mimi_pretrained_id": mimi_cfg.pretrained_id,
        "mimi_num_quantizers": mimi_cfg.num_quantizers,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare_cache(config: ExperimentConfig, device: str = "cpu") -> dict[str, Path]:
    """Extract and cache all configured train/validation/test splits.

    Skips any role whose cache directory already exists *and* was built
    from the same settings (checked via `_cache_fingerprint`). If the cache
    exists but was built with different settings, raises rather than
    silently training on stale data - delete the stale cache directory (and
    its `.fingerprint.json`) to force re-extraction.
    """
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
        fingerprint_path = cache_dir / f"{role}.fingerprint.json"
        out_paths[role] = out_path
        current_fp = _cache_fingerprint(config.data, config.mimi, specs)

        if out_path.exists():
            stored_fp = None
            if fingerprint_path.exists():
                stored_fp = json.loads(fingerprint_path.read_text()).get("fingerprint")
            if stored_fp == current_fp:
                logger.info("Cache for '%s' already exists and matches config, skipping.", role)
                continue
            raise RuntimeError(
                f"Cache for '{role}' at {out_path} does not match the current config "
                f"(stored fingerprint {stored_fp!r} != current {current_fp!r}); refusing to "
                "silently reuse it. Delete that directory (and "
                f"{fingerprint_path}) to re-extract with the current settings."
            )

        logger.info("Extracting '%s' split(s) %s ...", role, specs)
        dataset = extract_split(
            specs, config.data, config.mimi, device, role, drop_ctc_infeasible=(role == "train")
        )
        dataset.save_to_disk(str(out_path))
        fingerprint_path.write_text(json.dumps({"fingerprint": current_fp, "specs": specs}))
        logger.info("Saved '%s' cache to %s (%d examples).", role, out_path, len(dataset))
    return out_paths
