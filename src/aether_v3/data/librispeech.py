"""LibriSpeech loading via the `datasets` library.

Primary source is ``openslr/librispeech_asr`` (configs "clean"/"other", with
splits like "train.100", "train.360", "validation.clean", "test.other" -
see https://huggingface.co/datasets/openslr/librispeech_asr). That dataset
has had loading-script breakage on some `datasets` versions (see
huggingface/datasets issues #4179, #4609), so callers may configure a
fallback dataset id (e.g. ``distil-whisper/librispeech_asr``, an Arrow-native
mirror with no custom loading script) with matching split strings in
``DataConfig``.
"""
from __future__ import annotations

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def _parse_split_spec(spec: str) -> tuple[str, str]:
    """Parse "config_name/split_name" -> (config_name, split_name)."""
    config_name, _, split_name = spec.partition("/")
    if not split_name:
        raise ValueError(
            f"Invalid split spec '{spec}', expected '<config_name>/<split_name>' "
            "e.g. 'clean/train.100'."
        )
    return config_name, split_name


def load_split(spec: str, dataset_id: str, fallback_dataset_id: str | None = None) -> Dataset:
    """Load a single LibriSpeech split, e.g. spec="clean/train.100".

    Tries `dataset_id` first, falls back to `fallback_dataset_id` (same split
    spec string) if the primary source fails to load.
    """
    config_name, split_name = _parse_split_spec(spec)
    try:
        return load_dataset(dataset_id, config_name, split=split_name)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, external service failure
        if fallback_dataset_id is None:
            raise
        logger.warning(
            "Failed to load %s (%s) from %s: %s. Falling back to %s.",
            spec,
            split_name,
            dataset_id,
            exc,
            fallback_dataset_id,
        )
        return load_dataset(fallback_dataset_id, config_name, split=split_name)


def load_splits(
    specs: list[str], dataset_id: str, fallback_dataset_id: str | None = None
) -> Dataset:
    """Load and concatenate multiple LibriSpeech splits into one Dataset."""
    from datasets import concatenate_datasets

    parts = [load_split(spec, dataset_id, fallback_dataset_id) for spec in specs]
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)
