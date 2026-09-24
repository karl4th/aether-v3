"""Linear warmup + cosine decay LR schedule."""

from __future__ import annotations

import math

import torch


class WarmupPlateauScheduler:
    """Linear warmup, then hold and reduce LR on validation-WER plateaus."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        min_lr_ratio: float,
        factor: float,
        patience_evals: int,
        min_delta: float,
    ) -> None:
        if not 0.0 < factor < 1.0:
            raise ValueError("plateau_factor must be between zero and one")
        if patience_evals <= 0:
            raise ValueError("plateau_patience_evals must be positive")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.factor = factor
        self.patience_evals = patience_evals
        self.min_delta = min_delta
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self.best_metric = float("inf")
        self.bad_evals = 0
        self.reductions = 0
        self._set_multiplier(0.0 if warmup_steps > 0 else 1.0)

    def _set_multiplier(self, multiplier: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            self._set_multiplier(self.step_count / max(1, self.warmup_steps))

    def step_metric(self, metric: float) -> bool:
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.bad_evals = 0
            return False
        self.bad_evals += 1
        if self.bad_evals < self.patience_evals:
            return False
        changed = False
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            floor = base_lr * self.min_lr_ratio
            reduced = max(floor, group["lr"] * self.factor)
            changed |= reduced < group["lr"]
            group["lr"] = reduced
        self.bad_evals = 0
        if changed:
            self.reductions += 1
        return changed

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "best_metric": self.best_metric,
            "bad_evals": self.bad_evals,
            "reductions": self.reductions,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_count = int(state_dict["step_count"])
        self.best_metric = float(state_dict["best_metric"])
        self.bad_evals = int(state_dict["bad_evals"])
        self.reductions = int(state_dict["reductions"])


def _lr_multiplier(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
    *,
    schedule: str = "cosine",
    plateau_factor: float = 0.5,
    plateau_patience_evals: int = 3,
    plateau_min_delta: float = 0.005,
) -> torch.optim.lr_scheduler.LambdaLR | WarmupPlateauScheduler:
    if schedule == "plateau":
        return WarmupPlateauScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            factor=plateau_factor,
            patience_evals=plateau_patience_evals,
            min_delta=plateau_min_delta,
        )
    if schedule != "cosine":
        raise ValueError(f"unsupported lr_schedule: {schedule!r}")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(step, warmup_steps, max_steps, min_lr_ratio),
    )
