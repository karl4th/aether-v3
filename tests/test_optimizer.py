import torch

from aether_v3.config import AetherSpeechConfig, CTCConfig, TrainConfig
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.training.optimizer import build_optimizer


def test_optimizer_separates_decay_and_encoder_learning_rate():
    model = AetherCTCModel(
        AetherSpeechConfig(
            semantic_vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_size=16,
            dropout=0.0,
        ),
        CTCConfig(
            vocab_size=6,
            blank_id=5,
            upsample_factor=2,
            upsampler_num_layers=1,
            upsampler_num_heads=2,
            upsampler_ffn_size=16,
        ),
    )
    config = TrainConfig(lr=1e-3, encoder_lr_multiplier=0.5, weight_decay=0.1)
    optimizer = build_optimizer(model, config, torch.device("cpu"))

    combinations = {(group["lr"], group["weight_decay"]) for group in optimizer.param_groups}
    assert combinations == {(5e-4, 0.0), (5e-4, 0.1), (1e-3, 0.0), (1e-3, 0.1)}
