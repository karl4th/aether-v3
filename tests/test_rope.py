import pytest
import torch

from aether_v3.models.rope import apply_rope, build_rope_cache, rotate_half


def test_rope_cache_shapes_and_range():
    cos, sin = build_rope_cache(
        seq_len=10, head_dim=8, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32
    )
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)
    assert torch.all(cos <= 1.0 + 1e-6) and torch.all(cos >= -1.0 - 1e-6)
    assert torch.all(sin <= 1.0 + 1e-6) and torch.all(sin >= -1.0 - 1e-6)


def test_rope_at_position_zero_is_identity():
    # cos(0)=1, sin(0)=0 at every frequency -> rotation at position 0 must
    # leave the vector unchanged.
    cos, sin = build_rope_cache(
        seq_len=1, head_dim=4, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32
    )
    x = torch.randn(2, 3, 1, 4)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_offset_matches_slice_of_full_cache():
    full_cos, full_sin = build_rope_cache(
        seq_len=10, head_dim=8, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32
    )
    offset_cos, offset_sin = build_rope_cache(
        seq_len=4,
        head_dim=8,
        theta=10000.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position_offset=3,
    )
    assert torch.equal(offset_cos, full_cos[3:7])
    assert torch.equal(offset_sin, full_sin[3:7])


def test_rope_preserves_vector_norm():
    # RoPE is a rotation - it must not change each head vector's L2 norm.
    cos, sin = build_rope_cache(
        seq_len=16, head_dim=32, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32
    )
    x = torch.randn(4, 5, 16, 32)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_rotate_half_is_self_inverse_up_to_sign():
    x = torch.randn(2, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


def test_build_rope_cache_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        build_rope_cache(
            seq_len=4, head_dim=5, theta=10000.0, device=torch.device("cpu"), dtype=torch.float32
        )
