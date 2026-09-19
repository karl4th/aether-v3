import torch.nn as nn

from aether_v3.training import dist_utils


def test_defaults_without_torchrun_env(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert dist_utils.get_world_size() == 1
    assert dist_utils.get_rank() == 0
    assert dist_utils.get_local_rank() == 0
    assert dist_utils.is_distributed() is False
    assert dist_utils.is_main_process() is True


def test_is_distributed_true_when_world_size_gt_1(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert dist_utils.is_distributed() is True


def test_is_main_process_false_for_nonzero_rank(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    assert dist_utils.is_main_process() is False


def test_unwrap_model_returns_plain_module_unchanged():
    model = nn.Linear(2, 2)
    assert dist_utils.unwrap_model(model) is model
