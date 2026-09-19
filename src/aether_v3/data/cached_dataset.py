"""Reads the Arrow datasets produced by `mimi_cache.py` for training."""
from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class CTCCachedDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.dataset = load_from_disk(str(path))
        self.dataset.set_format("python")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]
        codes = torch.tensor(row["semantic_codes"], dtype=torch.long)
        target = torch.tensor(row["byte_target"], dtype=torch.long)
        return codes, target
