"""Optimizer construction with explicit decay and encoder LR groups."""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_v3.config import TrainConfig


def build_optimizer(
    model: nn.Module, config: TrainConfig, device: torch.device
) -> torch.optim.AdamW:
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        bare_name = name.removeprefix("module.")
        lr = config.lr * (config.encoder_lr_multiplier if bare_name.startswith("encoder.") else 1.0)
        # Biases, norm scales, and other vector parameters should not be decayed.
        weight_decay = config.weight_decay if parameter.ndim >= 2 else 0.0
        grouped.setdefault((lr, weight_decay), []).append(parameter)

    parameter_groups = [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), parameters in grouped.items()
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        fused=config.use_fused_adamw and device.type == "cuda",
    )
