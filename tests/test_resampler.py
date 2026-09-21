import torch

from aether_v3.config import ResamplerConfig
from aether_v3.models.resampler import AetherResampler


def _tiny_cfg(**overrides) -> ResamplerConfig:
    defaults = dict(enabled=True, ratio=4, num_heads=2, ffn_size=16, dropout=0.0, conv_kernel=5)
    defaults.update(overrides)
    return ResamplerConfig(**defaults)


def test_disabled_is_identity():
    resampler = AetherResampler(hidden_size=8, cfg=_tiny_cfg(enabled=False))
    x = torch.randn(2, 12, 8)
    mask = torch.ones(2, 12, dtype=torch.bool)
    out, out_mask = resampler(x, mask)
    assert out is x
    assert out_mask is mask


def test_ratio_one_is_identity():
    resampler = AetherResampler(hidden_size=8, cfg=_tiny_cfg(ratio=1))
    x = torch.randn(2, 12, 8)
    out, _ = resampler(x, None)
    assert out is x


def test_rejects_non_power_of_two_ratio():
    try:
        AetherResampler(hidden_size=8, cfg=_tiny_cfg(ratio=3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for ratio=3")


def test_ratio_four_shape_and_mask():
    resampler = AetherResampler(hidden_size=8, cfg=_tiny_cfg(ratio=4))
    x = torch.randn(2, 16, 8)
    # example 0 fully valid, example 1 only the first 8 frames real
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 8:] = False

    out, out_mask = resampler(x, mask)
    assert out.shape[0] == 2
    assert out.shape[2] == 8
    assert out_mask.shape == out.shape[:2]
    # example 1 (half the input length) should end up with roughly half the
    # valid output length of example 0.
    assert int(out_mask[1].sum()) < int(out_mask[0].sum())


def test_ratio_four_backward():
    resampler = AetherResampler(hidden_size=8, cfg=_tiny_cfg(ratio=4))
    x = torch.randn(2, 16, 8, requires_grad=True)
    mask = torch.ones(2, 16, dtype=torch.bool)
    out, _ = resampler(x, mask)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)
