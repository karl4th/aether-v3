import math

import torch

from aether_v3.training.scheduler import WarmupPlateauScheduler, _lr_multiplier, build_scheduler


def test_warmup_ramps_linearly():
    assert _lr_multiplier(0, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1) == 0.0
    assert _lr_multiplier(50, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1) == 0.5


def test_peak_at_end_of_warmup():
    mult = _lr_multiplier(100, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert math.isclose(mult, 1.0, abs_tol=1e-6)


def test_cosine_decay_reaches_min_ratio_at_max_steps():
    mult = _lr_multiplier(1000, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert math.isclose(mult, 0.1, abs_tol=1e-6)


def test_multiplier_stays_within_bounds():
    # Warmup intentionally ramps from 0 (below min_lr_ratio) up to the peak;
    # only *after* warmup is min_lr_ratio a real floor.
    warmup_steps = 100
    for step in range(0, 1200, 17):
        mult = _lr_multiplier(step, warmup_steps=warmup_steps, max_steps=1000, min_lr_ratio=0.1)
        lower_bound = 0.1 - 1e-6 if step >= warmup_steps else -1e-6
        assert lower_bound <= mult <= 1.0 + 1e-6


def test_build_scheduler_reaches_peak_lr_at_end_of_warmup():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = build_scheduler(optimizer, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert math.isclose(optimizer.param_groups[0]["lr"], 1.0, abs_tol=1e-6)


def test_plateau_scheduler_holds_then_reduces_after_patience():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = WarmupPlateauScheduler(
        optimizer,
        warmup_steps=2,
        min_lr_ratio=0.1,
        factor=0.5,
        patience_evals=3,
        min_delta=0.01,
    )
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert not scheduler.step_metric(0.8)
    assert not scheduler.step_metric(0.795)
    assert not scheduler.step_metric(0.794)
    assert scheduler.step_metric(0.793)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_build_scheduler_rejects_unknown_schedule():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with __import__("pytest").raises(ValueError, match="unsupported lr_schedule"):
        build_scheduler(optimizer, 10, 100, 0.1, schedule="unknown")
