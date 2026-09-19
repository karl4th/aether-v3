import torch

from aether_v3.training.checkpoint import load_checkpoint, save_checkpoint


def _make_triple(lr: float = 0.1):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    return model, optimizer, scheduler


def test_save_and_load_roundtrip(tmp_path):
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scheduler, step=42, best_cer=0.5)

    new_model, new_optimizer, new_scheduler = _make_triple()
    ckpt = load_checkpoint(path, new_model, new_optimizer, new_scheduler)

    assert ckpt["step"] == 42
    assert ckpt["best_cer"] == 0.5
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_load_without_optimizer_or_scheduler(tmp_path):
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scheduler, step=1, best_cer=1.0)

    new_model = torch.nn.Linear(4, 2)
    ckpt = load_checkpoint(path, new_model)
    assert ckpt["step"] == 1
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_save_creates_parent_directories(tmp_path):
    model, optimizer, scheduler = _make_triple()
    path = tmp_path / "nested" / "dir" / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scheduler, step=0, best_cer=float("inf"))
    assert path.exists()
