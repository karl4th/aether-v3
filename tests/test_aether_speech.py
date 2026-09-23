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


def test_future_frames_do_not_change_past_outputs():
    cfg = _tiny_config()
    encoder = AetherSpeechEncoder(cfg).eval()
    prefix = torch.randint(0, cfg.semantic_vocab_size, (1, 5))
    suffix = torch.randint(0, cfg.semantic_vocab_size, (1, 4))

    with torch.no_grad():
        prefix_out = encoder(prefix)
        full_out = encoder(torch.cat((prefix, suffix), dim=1))

    assert torch.allclose(prefix_out, full_out[:, : prefix.shape[1]], atol=1e-5)


@pytest.mark.parametrize("chunk_sizes", [(1,) * 9, (2, 3, 4), (4, 1, 4)])
def test_streaming_chunks_match_full_causal_forward(chunk_sizes):
    cfg = _tiny_config()
    encoder = AetherSpeechEncoder(cfg).eval()
    codes = torch.randint(0, cfg.semantic_vocab_size, (2, sum(chunk_sizes)))

    with torch.no_grad():
        full_out = encoder(codes)
        state = encoder.init_streaming_state()
        chunk_outputs = []
        start = 0
        for size in chunk_sizes:
            chunk_out, state = encoder.forward_chunk(codes[:, start : start + size], state)
            chunk_outputs.append(chunk_out)
            start += size

    streaming_out = torch.cat(chunk_outputs, dim=1)
    assert torch.allclose(full_out, streaming_out, atol=1e-5)
    assert state.frames_seen == codes.shape[1]


def test_streaming_cache_is_bounded():
    cfg = _tiny_config()
    cfg.streaming_left_context_frames = 4
    encoder = AetherSpeechEncoder(cfg).eval()
    state = encoder.init_streaming_state()

    with torch.no_grad():
        for _ in range(6):
            _, state = encoder.forward_chunk(
                torch.randint(0, cfg.semantic_vocab_size, (1, 2)), state
            )

    assert state.frames_seen == 12
    for cache in state.layer_caches:
        assert cache is not None
        assert cache.key.shape[2] == cfg.streaming_left_context_frames
        assert cache.value.shape[2] == cfg.streaming_left_context_frames
        assert cache.key_mask.shape[1] == cfg.streaming_left_context_frames


def test_streaming_matches_full_forward_after_context_window_rolls():
    cfg = _tiny_config()
    cfg.streaming_left_context_frames = 4
    encoder = AetherSpeechEncoder(cfg).eval()
    codes = torch.randint(0, cfg.semantic_vocab_size, (1, 12))

    with torch.no_grad():
        full_out = encoder(codes)
        state = encoder.init_streaming_state()
        outputs = []
        for chunk in codes.split(2, dim=1):
            output, state = encoder.forward_chunk(chunk, state)
            outputs.append(output)

    assert torch.allclose(full_out, torch.cat(outputs, dim=1), atol=1e-5)


def test_streaming_rejects_padded_chunks():
    encoder = AetherSpeechEncoder(_tiny_config())
    codes = torch.ones(2, 3, dtype=torch.long)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    with pytest.raises(ValueError, match="cannot contain padded frames"):
        encoder.forward_chunk(codes, attention_mask=mask)


def test_flush_emits_nothing_and_resets_state():
    encoder = AetherSpeechEncoder(_tiny_config()).eval()
    with torch.no_grad():
        _, state = encoder.forward_chunk(torch.ones(1, 3, dtype=torch.long))

    delayed, reset_state = encoder.flush_stream(state)

    assert delayed is None
    assert reset_state.frames_seen == 0
    assert all(cache is None for cache in reset_state.layer_caches)


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
