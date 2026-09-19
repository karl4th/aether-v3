"""Linear warmup + cosine decay LR schedule."""

from __future__ import annotations

import math

import torch


def _lr_multiplier(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(step, warmup_steps, max_steps, min_lr_ratio),
    )
