"""Single-GPU / multi-GPU DDP helpers.

Detects `torchrun`'s environment variables (`WORLD_SIZE`, `RANK`,
`LOCAL_RANK`) to decide whether to run distributed. Absent those (plain
`python -m ...`), everything here is a no-op and training runs single
process on one GPU (or CPU).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> torch.device:
    if is_distributed():
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def teardown_distributed() -> None:
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if is_distributed():
        device_ids = [device.index] if device.type == "cuda" else None
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
