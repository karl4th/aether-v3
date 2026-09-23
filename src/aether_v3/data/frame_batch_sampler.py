"""Deterministic length-bucketed batches bounded by semantic-frame budget."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class FrameBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_frames: int,
        max_examples: int,
        shuffle: bool,
        seed: int,
        bucket_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("lengths must be non-empty positive integers")
        if max_frames <= 0 or max_examples <= 0 or bucket_size <= 0:
            raise ValueError("batch limits and bucket_size must be positive")
        if max(lengths) > max_frames:
            raise ValueError(
                f"longest sample ({max(lengths)} frames) exceeds batch budget ({max_frames})"
            )
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        self.lengths = tuple(int(length) for length in lengths)
        self.max_frames = max_frames
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            current: list[int] = []
            current_max = 0
            for index in bucket:
                candidate_max = max(current_max, self.lengths[index])
                candidate_size = len(current) + 1
                if current and (
                    candidate_size > self.max_examples
                    or candidate_max * candidate_size > self.max_frames
                ):
                    batches.append(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, self.lengths[index])
            if current and (not self.drop_last or len(current) == self.max_examples):
                batches.append(current)

        if self.shuffle:
            rng.shuffle(batches)
        if self.world_size > 1:
            usable = len(batches) - len(batches) % self.world_size
            batches = batches[:usable]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._global_batches()[self.rank :: self.world_size]

    def __len__(self) -> int:
        count = len(self._global_batches())
        return math.ceil(count / self.world_size)
