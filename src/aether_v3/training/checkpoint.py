"""Atomic model snapshots and complete resumable training checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aether_v3.training.dist_utils import unwrap_model

SCHEMA_VERSION = 2


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_model_weights(path: str | Path, model: nn.Module, *, step: int) -> None:
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "model_weights",
            "model": unwrap_model(model).state_dict(),
            "step": step,
        },
        path,
    )


def load_model_weights(
    path: str | Path, model: nn.Module, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    snapshot = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = snapshot.get("model", snapshot)
    unwrap_model(model).load_state_dict(state_dict)
    return snapshot


def load_encoder_weights(
    path: str | Path, encoder: nn.Module, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Initialize only AetherSpeech from an encoder or full Stage 1 snapshot."""
    snapshot = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = snapshot.get("model", snapshot)
    encoder_prefixes = ("encoder.", "module.encoder.")
    encoder_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        for prefix in encoder_prefixes
        if key.startswith(prefix)
    }
    if not encoder_state:
        encoder_state = state_dict
    encoder.load_state_dict(encoder_state)
    return snapshot


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    epoch: int,
    batches_in_epoch: int,
    best_metrics: dict[str, float],
    termination_reason: str | None = None,
) -> None:
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "training_state",
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "batches_in_epoch": batches_in_epoch,
            "best_metrics": best_metrics,
            "rng_state": capture_rng_state(),
            "termination_reason": termination_reason,
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("kind") != "training_state"
    ):
        raise ValueError(
            f"checkpoint kind/schema {checkpoint.get('kind')!r}/"
            f"{checkpoint.get('schema_version')!r} is not resumable "
            f"by schema {SCHEMA_VERSION}; use it only as init_encoder_from"
        )
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
