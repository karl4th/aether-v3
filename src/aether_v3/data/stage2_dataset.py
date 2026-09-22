"""Reads the `.pt` caches produced by `stage2_cache.build_stage2_cache`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from aether_v3.data.stage2_cache import load_stage2_cache


class Stage2CachedDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.records: list[dict[str, Any]] = load_stage2_cache(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class Stage2ShardDataset(IterableDataset):
    """Streams shuffled records from independently loadable ``shard-*.pt`` files."""

    def __init__(self, directory: str | Path, shuffle: bool = True, seed: int = 1337) -> None:
        self.paths = sorted(Path(directory).glob("shard-*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No shard-*.pt files under {directory}")
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import random

        worker = get_worker_info()
        paths = self.paths[worker.id :: worker.num_workers] if worker else self.paths
        rng = random.Random(self.seed + self.epoch + (worker.id if worker else 0))
        paths = list(paths)
        if self.shuffle:
            rng.shuffle(paths)
        for path in paths:
            records = load_stage2_cache(path)
            if self.shuffle:
                rng.shuffle(records)
            yield from records
