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
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch.utils.data
import torchaudio
from datasets import Dataset

from aether_v3.config import DataConfig, ExperimentConfig, MimiConfig
from aether_v3.data.librispeech import load_splits
from aether_v3.data.tokenizer import text_to_byte_ids
from aether_v3.models.mimi_wrapper import FrozenMimi

logger = logging.getLogger(__name__)


class _RawAudioDataset(torch.utils.data.Dataset):
    """Thin torch Dataset over a raw (undedecoded-until-accessed) HF split.

    Exists so audio decode+resample (CPU-bound: FLAC decode via
    soundfile/librosa, then resampling to Mimi's target rate) can run in
    `DataLoader` worker processes, overlapping with Mimi's GPU encode in the
    main process instead of blocking it - without this, extraction was
    measured at ~13 examples/sec on a Colab GPU session because decode ran
    serially before every GPU call. Resample in particular used to happen
    in `FrozenMimi.encode_semantic` itself (main process, one waveform at a
    time, ahead of every batch's GPU call) - moved here so it's parallel
    across workers instead of serially stalling the GPU each batch.
    """

    def __init__(self, raw: Dataset, sample_rate_in: int, target_sample_rate: int, role: str) -> None:
        self.raw = raw
        self.sample_rate_in = sample_rate_in
        self.target_sample_rate = target_sample_rate
        self.role = role

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        row = self.raw[idx]
        audio = row["audio"]
        sr = audio["sampling_rate"]
        if sr != self.sample_rate_in:
            raise ValueError(
                f"Unexpected sample rate {sr} Hz for {self.role} split "
                f"(configured sample_rate_in={self.sample_rate_in})."
            )
        waveform = np.asarray(audio["array"], dtype=np.float32)
        if self.sample_rate_in != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), self.sample_rate_in, self.target_sample_rate
            ).numpy()
        return waveform, row["text"]


def _encode_with_oom_retry(mimi: FrozenMimi, waveforms: list[np.ndarray]) -> list[np.ndarray]:
    """`mimi.encode_semantic`, halving the batch and retrying on CUDA OOM.

    A single unlucky batch (several near-`max_audio_seconds` clips padded
    together) can exceed VRAM even when the configured batch size normally
    fits comfortably. Without this, that one batch would crash the entire
    `extract_split()` generator - and since `prepare_cache` only writes the
    split to disk after it finishes completely, every example already
    processed in this run would be lost too. Halving is a one-off recovery
    for a rare bad batch, not a substitute for a sane `extraction_batch_size`.
    """
    try:
        return mimi.encode_semantic(waveforms)
    except torch.cuda.OutOfMemoryError:
        if len(waveforms) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(waveforms) // 2
        logger.warning(
            "CUDA OOM on a batch of %d waveforms - splitting into %d + %d and retrying.",
            len(waveforms),
            mid,
            len(waveforms) - mid,
        )
        return _encode_with_oom_retry(mimi, waveforms[:mid]) + _encode_with_oom_retry(
            mimi, waveforms[mid:]
        )


def _ctc_min_input_length(target_ids: list[int]) -> int:
    """Minimum number of input frames CTC needs to align `target_ids`.

    One frame per label, plus one extra frame between any two adjacent
    identical labels (a blank must separate repeated consecutive labels, or
    they'd collapse into one during decoding).
    """
    if not target_ids:
        return 0
    # Deliberately mismatched lengths (adjacent-pair iteration).
    repeats = sum(1 for a, b in zip(target_ids, target_ids[1:], strict=False) if a == b)
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
    n_consumed = 0

    # `datasets`' own progress bar for this generator can't show a total
    # (Dataset.from_generator doesn't know the split size ahead of time,
    # hence the "N/0" it prints) - log our own, since `n_total` is known
    # here, roughly every 2% of the split so it doesn't spam small splits.
    log_every = max(1, n_total // 50)
    t_start = time.time()
    logger.info(
        "%s: starting extraction of %d examples (batch_size=%d, num_workers=%d)",
        role,
        n_total,
        data_cfg.extraction_batch_size,
        data_cfg.extraction_num_workers,
    )

    raw_loader = torch.utils.data.DataLoader(
        _RawAudioDataset(raw, data_cfg.sample_rate_in, mimi.target_sample_rate, role),
        batch_size=data_cfg.extraction_batch_size,
        shuffle=False,
        num_workers=data_cfg.extraction_num_workers,
        collate_fn=list,
        prefetch_factor=4 if data_cfg.extraction_num_workers > 0 else None,
    )
    for batch in raw_loader:
        waveforms: list[np.ndarray] = []
        texts: list[str] = []
        for wav, text in batch:
            # wav is already resampled to mimi.target_sample_rate by
            # _RawAudioDataset - length must be measured against that rate,
            # not the source sample_rate_in.
            duration = len(wav) / mimi.target_sample_rate
            if not (data_cfg.min_audio_seconds <= duration <= data_cfg.max_audio_seconds):
                n_dropped_duration += 1
                continue
            waveforms.append(wav)
            texts.append(text)

        n_consumed += len(batch)
        if n_consumed % log_every < len(batch):
            elapsed = time.time() - t_start
            rate = n_consumed / elapsed if elapsed > 0 else 0.0
            remaining = (n_total - n_consumed) / rate if rate > 0 else float("inf")
            logger.info(
                "%s: %d/%d (%.1f%%) - %.0fs elapsed, ~%.0fs remaining (%.1f examples/s)",
                role,
                n_consumed,
                n_total,
                100.0 * n_consumed / n_total,
                elapsed,
                remaining,
                rate,
            )

        if not waveforms:
            continue

        code_seqs = _encode_with_oom_retry(mimi, waveforms)
        for codes, text in zip(code_seqs, texts, strict=True):
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


def _role_to_specs(config: ExperimentConfig) -> dict[str, list[str]]:
    return {
        "train": config.data.train_splits,
        "validation": config.data.validation_splits,
        "test": config.data.test_splits,
    }


def download_raw_splits(config: ExperimentConfig) -> None:
    """Downloads and builds every configured LibriSpeech split via HF `datasets`,
    without touching Mimi at all.

    Meant to be run as its own step (its own notebook cell, ideally) ahead of
    `prepare_cache`: HF's own download/build progress bars show a real total
    (e.g. "28539/28539"), but the Mimi-extraction generator that follows
    can't (`Dataset.from_generator` doesn't know a generator's length ahead
    of time, hence the "N/0" progress it shows during that stage) - running
    these as two separate calls makes it unambiguous which one is actually
    in progress, instead of two different "N/..." bars that look identical
    at a glance. `prepare_cache` calls `load_splits` again per role, but
    once downloaded/built here, that's a instant local-cache hit, not a
    second download.
    """
    for role, specs in _role_to_specs(config).items():
        logger.info("Downloading '%s' split(s) %s ...", role, specs)
        load_splits(specs, config.data.dataset_id, config.data.fallback_dataset_id)
        logger.info("'%s' split(s) %s ready in the local HF datasets cache.", role, specs)


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

    role_to_specs = _role_to_specs(config)
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
