"""Pinned LoquaciousSet Mimi-Parquet access for Stage 1 training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from aether_v3.config import DataConfig
from aether_v3.data.tokenizer import text_to_byte_ids


class LoquaciousSemanticDataset(Dataset):
    """Adapts published q0 rows to the tensors consumed by Stage 1."""

    REQUIRED_COLUMNS = frozenset(
        {"semantic_codes", "semantic_length", "normalized_text", "sample_id", "audio_seconds"}
    )

    def __init__(self, dataset: HFDataset) -> None:
        missing = self.REQUIRED_COLUMNS.difference(dataset.column_names)
        if missing:
            raise ValueError(f"Loquacious cache is missing columns: {sorted(missing)}")
        self.dataset = dataset
        # Reading one primitive Arrow column is cheap and gives the sampler
        # exact lengths without materializing semantic token sequences.
        self.lengths = tuple(int(length) for length in dataset["semantic_length"])
        if any(length <= 0 for length in self.lengths):
            raise ValueError("semantic_length must be positive for every row")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        row = self.dataset[int(idx)]
        codes = row["semantic_codes"]
        if len(codes) != self.lengths[int(idx)]:
            raise ValueError(f"semantic_length mismatch for sample {row['sample_id']}")
        target = text_to_byte_ids(row["normalized_text"])
        if not target:
            raise ValueError(f"empty normalized_text for sample {row['sample_id']}")
        return (
            torch.tensor(codes, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            float(row["audio_seconds"]),
        )


def load_loquacious_split(config: DataConfig, split: str) -> LoquaciousSemanticDataset:
    """Load one split from an immutable Hub revision using HF_TOKEN implicitly."""
    if not config.dataset_revision:
        raise ValueError("data.dataset_revision is required for hf_parquet training")
    dataset = load_dataset(
        config.dataset_id,
        split=split,
        revision=config.dataset_revision,
        cache_dir=str(Path(config.cache_dir)),
    )
    if not isinstance(dataset, HFDataset):
        raise TypeError(f"expected a map-style Dataset for split {split!r}")
    return LoquaciousSemanticDataset(dataset)


def dataset_lengths(dataset: Dataset) -> Sequence[int] | None:
    lengths = getattr(dataset, "lengths", None)
    return lengths if isinstance(lengths, Sequence) else None
