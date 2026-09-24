from typing import Any

import torch

from aether_v3.config import AetherSpeechConfig, CTCConfig
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.models.ctc_head import compute_ctc_loss


def _tiny_ctc_cfg(**overrides) -> CTCConfig:
    # upsampler_num_heads=2 to match the tiny hidden_size=8 speech config
    # used in these tests (8 isn't divisible by the default 12 heads).
    defaults: dict[str, Any] = dict(
        vocab_size=6,
        blank_id=5,
        upsample_factor=2,
        upsampler_num_layers=1,
        upsampler_num_heads=2,
        upsampler_ffn_size=16,
        upsampler_dropout=0.0,
    )
    defaults.update(overrides)
    return CTCConfig(**defaults)


def test_end_to_end_forward_and_backward():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=2, num_heads=2, ffn_size=16, dropout=0.0
    )
    ctc_cfg = _tiny_ctc_cfg()
    model = AetherCTCModel(speech_cfg, ctc_cfg)

    codes = torch.randint(0, speech_cfg.semantic_vocab_size, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.bool)
    log_probs = model(codes, mask)
    assert log_probs.shape == (2, 12 * ctc_cfg.upsample_factor, ctc_cfg.vocab_size)

    targets = torch.tensor([0, 1, 2, 3])
    input_lengths = torch.tensor([12, 12]) * ctc_cfg.upsample_factor
    target_lengths = torch.tensor([2, 2])
    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths, ctc_cfg.blank_id)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_ctc_logits_do_not_depend_on_future_semantic_frames():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    model = AetherCTCModel(speech_cfg, _tiny_ctc_cfg()).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    first = torch.cat((prefix, torch.tensor([[5, 6, 7, 8]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[9, 10, 11, 12]])), dim=1)
    mask = torch.ones_like(first, dtype=torch.bool)

    with torch.no_grad():
        first_logits = model(first, mask)
        second_logits = model(second, mask)

    prefix_steps = prefix.shape[1] * model.upsample_factor
    torch.testing.assert_close(
        first_logits[:, :prefix_steps],
        second_logits[:, :prefix_steps],
        atol=1e-6,
        rtol=1e-6,
    )


def test_ctc_lookahead_is_bounded_to_configured_semantic_frames():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        lookahead_frames=2,
    )
    model = AetherCTCModel(speech_cfg, _tiny_ctc_cfg(upsampler_num_layers=2)).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[:, 7] = 9
    mask = torch.ones_like(first, dtype=torch.bool)

    with torch.no_grad():
        first_logits = model(first, mask)
        second_logits = model(second, mask)

    # Frame zero may see only through semantic frame two. A change in frame
    # seven must therefore leave all of frame zero's subframes unchanged.
    torch.testing.assert_close(
        first_logits[:, : model.upsample_factor],
        second_logits[:, : model.upsample_factor],
        atol=1e-6,
        rtol=1e-6,
    )


def test_total_lookahead_does_not_compound_across_depth():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=32,
        hidden_size=8,
        num_layers=3,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        lookahead_frames=2,
    )
    model = AetherCTCModel(
        speech_cfg, _tiny_ctc_cfg(upsampler_num_layers=2)
    ).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = torch.ones_like(original, dtype=torch.bool)

    with torch.no_grad():
        baseline = model(original, mask)
        at_budget = original.clone()
        at_budget[:, 2] = 7
        within = model(at_budget, mask)
        beyond_budget = original.clone()
        beyond_budget[:, 3] = 8
        beyond = model(beyond_budget, mask)

    first_frame = model.upsample_factor
    assert not torch.allclose(baseline[:, :first_frame], within[:, :first_frame])
    torch.testing.assert_close(
        baseline[:, :first_frame], beyond[:, :first_frame], atol=1e-6, rtol=1e-6
    )


def test_full_ctc_path_matches_chunked_streaming_with_lookahead():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=32,
        hidden_size=8,
        num_layers=3,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        streaming_left_context_frames=64,
        lookahead_frames=2,
    )
    model = AetherCTCModel(
        speech_cfg, _tiny_ctc_cfg(upsampler_num_layers=2)
    ).eval()
    codes = torch.randint(0, speech_cfg.semantic_vocab_size, (1, 11))
    mask = torch.ones_like(codes, dtype=torch.bool)

    with torch.no_grad():
        full = model(codes, mask)
        encoder_state = None
        upsampler_state = None
        pieces = []
        start = 0
        for size in (1, 3, 2, 5):
            logits, encoder_state, upsampler_state = model.forward_chunk(
                codes[:, start : start + size], encoder_state, upsampler_state
            )
            pieces.append(logits)
            start += size
        assert encoder_state is not None
        tail, _, _ = model.flush_stream(encoder_state, upsampler_state)
        assert tail is not None
        pieces.append(tail)

    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)


def test_joint_objective_backpropagates_into_encoder_and_semantic_heads():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    model = AetherCTCModel(speech_cfg, _tiny_ctc_cfg())
    codes = torch.randint(0, speech_cfg.semantic_vocab_size, (2, 12))
    mask = torch.ones_like(codes, dtype=torch.bool)

    outputs = model(codes, mask, compute_semantic_loss=True)
    assert isinstance(outputs, tuple)
    log_probs, semantic_loss = outputs
    ctc_loss = compute_ctc_loss(
        log_probs,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([24, 24]),
        torch.tensor([2, 2]),
        blank_id=5,
    )
    (ctc_loss + model.semantic_prediction_weight * semantic_loss).backward()

    assert semantic_loss.item() > 0
    assert model.encoder.semantic_embedding.weight.grad is not None
    assert model.semantic_prediction is not None
    assert model.semantic_prediction.heads[0].weight.grad is not None


def test_semantic_prediction_ignores_padded_targets():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    model = AetherCTCModel(speech_cfg, _tiny_ctc_cfg()).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    padded_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    padded_b = torch.tensor([[1, 2, 3, 4, 9, 10]])
    mask = torch.tensor([[True, True, True, True, False, False]])

    with torch.no_grad():
        _, hidden_a = model.forward_with_hidden(padded_a, mask)
        _, hidden_b = model.forward_with_hidden(padded_b, mask)
        loss_a = model.semantic_loss(hidden_a, padded_a, mask)
        loss_b = model.semantic_loss(hidden_b, padded_b, mask)
        _, hidden_prefix = model.forward_with_hidden(
            prefix, torch.ones_like(prefix, dtype=torch.bool)
        )
        loss_prefix = model.semantic_loss(
            hidden_prefix, prefix, torch.ones_like(prefix, dtype=torch.bool)
        )

    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(loss_a, loss_prefix)
