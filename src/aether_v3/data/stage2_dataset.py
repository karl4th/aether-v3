"""Reads the `.pt` caches produced by `stage2_cache.build_stage2_cache`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from aether_v3.data.stage2_cache import load_stage2_cache


class Stage2CachedDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.records: list[dict[str, Any]] = load_stage2_cache(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]
