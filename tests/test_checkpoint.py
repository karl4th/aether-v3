import random

import numpy as np
import pytest
import torch

from aether_v3.training.checkpoint import (
    load_model_weights,
    load_training_checkpoint,
    save_model_weights,
    save_training_checkpoint,
)


def _make_triple(lr: float = 0.1):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    return model, optimizer, scheduler


def test_training_checkpoint_roundtrip(tmp_path):
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        step=42,
        epoch=3,
        batches_in_epoch=7,
        best_metrics={"eval_wer": 0.2},
    )

    new_model, new_optimizer, new_scheduler = _make_triple()
    checkpoint = load_training_checkpoint(path, new_model, new_optimizer, new_scheduler)

    assert checkpoint["step"] == 42
    assert checkpoint["epoch"] == 3
    assert checkpoint["batches_in_epoch"] == 7
    assert checkpoint["best_metrics"] == {"eval_wer": 0.2}
    for expected, actual in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_model_snapshot_loads_without_optimizer(tmp_path):
    model, _, _ = _make_triple()
    path = tmp_path / "model.pt"
    save_model_weights(path, model, step=5)

    new_model = torch.nn.Linear(4, 2)
    snapshot = load_model_weights(path, new_model)

    assert snapshot["step"] == 5
    for expected, actual in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_save_is_atomic_and_leaves_no_temporary_file(tmp_path):
    model, _, _ = _make_triple()
    path = tmp_path / "nested" / "model.pt"
    save_model_weights(path, model, step=0)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()


def test_resume_rejects_weights_only_snapshot(tmp_path):
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "model.pt"
    save_model_weights(path, model, step=1)

    with pytest.raises(ValueError, match="not resumable"):
        load_training_checkpoint(path, model, optimizer, scheduler)


def test_training_checkpoint_restores_rng_states(tmp_path):
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        step=1,
        epoch=0,
        batches_in_epoch=1,
        best_metrics={},
    )
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(20)
    np.random.seed(21)
    torch.manual_seed(22)

    load_training_checkpoint(path, model, optimizer, scheduler)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert actual == expected
