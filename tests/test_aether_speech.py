import pytest
import torch

from aether_v3.config import AetherSpeechConfig
from aether_v3.models.aether_speech import AetherSpeechEncoder


def _tiny_config() -> AetherSpeechConfig:
    return AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )


def test_forward_shape():
    cfg = _tiny_config()
    encoder = AetherSpeechEncoder(cfg)
    codes = torch.randint(0, cfg.semantic_vocab_size, (3, 7))
    mask = torch.ones(3, 7, dtype=torch.bool)
    out = encoder(codes, mask)
    assert out.shape == (3, 7, cfg.hidden_size)


def test_padding_does_not_leak_into_valid_positions():
    """Two inputs, identical except for padded tail content/length, must
    produce identical hidden states at the valid (unpadded) positions -
    otherwise the attention mask isn't actually excluding padded keys."""
    cfg = _tiny_config()
    encoder = AetherSpeechEncoder(cfg)
    encoder.eval()

    torch.manual_seed(0)
    codes_a = torch.randint(0, cfg.semantic_vocab_size, (1, 5))
    pad = torch.randint(0, cfg.semantic_vocab_size, (1, 3))
    codes_b = torch.cat([codes_a, pad], dim=1)

    mask_a = torch.ones(1, 5, dtype=torch.bool)
    mask_b = torch.cat(
        [torch.ones(1, 5, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        out_a = encoder(codes_a, mask_a)
        out_b = encoder(codes_b, mask_b)

    assert torch.allclose(out_a, out_b[:, :5, :], atol=1e-5)


def test_rejects_hidden_size_not_divisible_by_heads():
    cfg = AetherSpeechConfig(hidden_size=10, num_heads=3)
    with pytest.raises(ValueError):
        AetherSpeechEncoder(cfg)


def test_gradients_flow_through_embedding_and_all_layers():
    cfg = _tiny_config()
    encoder = AetherSpeechEncoder(cfg)
    codes = torch.randint(0, cfg.semantic_vocab_size, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = encoder(codes, mask)
    out.sum().backward()
    assert encoder.semantic_embedding.weight.grad is not None
    assert torch.any(encoder.semantic_embedding.weight.grad != 0)
    for block in encoder.blocks:
        assert block.attn.q_proj.weight.grad is not None
