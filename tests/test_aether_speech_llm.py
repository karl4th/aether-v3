"""Structural tests for AetherSpeechLLM using a tiny, randomly-initialized
Qwen3 model (injected via the `llm=` constructor arg) instead of a real
pretrained download - fast, network-free checks of the
prefix/speech/target assembly, label masking, and frozen-LLM gradient
flow described in docs/stage2_spec.md Sec.15-16, 33, 47-48.
"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from aether_v3.config import (
    AetherSpeechConfig,
    BridgeConfig,
    ConnectorConfig,
    LLMConfig,
    ResamplerConfig,
)
from aether_v3.models.aether_speech_llm import AetherSpeechLLM

LLM_HIDDEN = 20


def _tiny_llm() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=LLM_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=10,
        max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config)


def _build_model(ratio: int = 1) -> AetherSpeechLLM:
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=2, num_heads=2, ffn_size=16, dropout=0.0
    )
    connector_cfg = ConnectorConfig(
        resampler=ResamplerConfig(
            enabled=ratio > 1, ratio=ratio, num_heads=2, ffn_size=16, dropout=0.0
        ),
        bridge=BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=LLM_HIDDEN, dropout=0.0),
    )
    llm_cfg = LLMConfig(model_id="unused", frozen=True, dtype="float32")
    return AetherSpeechLLM(speech_cfg, connector_cfg, llm_cfg, speech_frozen=True, llm=_tiny_llm())


def _batch() -> dict[str, torch.Tensor]:
    batch_size = 2
    semantic_codes = torch.randint(0, 16, (batch_size, 12))
    speech_mask = torch.ones(batch_size, 12, dtype=torch.bool)
    speech_mask[1, 9:] = False

    prefix_ids = torch.randint(0, 64, (batch_size, 3))
    prefix_mask = torch.ones(batch_size, 3, dtype=torch.bool)

    target_ids = torch.randint(0, 64, (batch_size, 4))
    target_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    target_mask[0, 3:] = False  # example 0 has a shorter real target

    return {
        "semantic_codes": semantic_codes,
        "speech_attention_mask": speech_mask,
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
    }


def test_forward_produces_finite_loss():
    model = _build_model()
    out = model(_batch())
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_labels_only_cover_target_span():
    model = _build_model()
    batch = _batch()
    speech_embeds, speech_mask = model.encode_speech(
        batch["semantic_codes"], batch["speech_attention_mask"]
    )
    built = model.build_inputs_embeds(
        batch["prefix_ids"],
        batch["prefix_mask"],
        speech_embeds,
        speech_mask,
        batch["target_ids"],
        batch["target_mask"],
    )
    for i in range(2):
        p_len = int(batch["prefix_mask"][i].sum())
        s_len = int(batch["speech_attention_mask"][i].sum())
        t_len = int(batch["target_mask"][i].sum())
        target_start = p_len + 1 + s_len + 1
        row = built.labels[i]
        assert torch.all(row[:target_start] == -100)
        assert torch.equal(row[target_start : target_start + t_len], batch["target_ids"][i, :t_len])
        assert torch.all(row[target_start + t_len :] == -100)


def test_frozen_llm_has_no_grad_but_connector_does():
    model = _build_model()
    out = model(_batch())
    out.loss.backward()

    for p in model.llm.parameters():
        assert p.grad is None

    connector_grads = [p.grad for p in model.connector.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in connector_grads)


def test_resampled_speech_still_flows_gradient_to_connector():
    model = _build_model(ratio=2)
    out = model(_batch())
    out.loss.backward()
    resampler_grads = [p.grad for p in model.connector.resampler.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in resampler_grads)


def test_forward_cached_matches_forward_from_raw_codes():
    """`forward_cached` (pre-computed encoder states, Stage 2 data pipeline
    v0) must be equivalent to `forward` (raw codes) run through the same
    frozen encoder - both should produce the same loss for the same
    underlying speech."""
    model = _build_model()
    batch = _batch()

    speech_states, speech_mask = model.encode_speech(
        batch["semantic_codes"], batch["speech_attention_mask"]
    )
    # encode_speech already runs the Connector - forward_cached expects
    # pre-*encoder* (not pre-connector) states, so undo that for the test
    # by calling the encoder directly instead.
    with torch.no_grad():
        raw_encoder_states = model.encoder(batch["semantic_codes"], batch["speech_attention_mask"])

    cached_batch = {
        "speech_states": raw_encoder_states,
        "speech_mask": batch["speech_attention_mask"],
        "prefix_ids": batch["prefix_ids"],
        "prefix_mask": batch["prefix_mask"],
        "target_ids": batch["target_ids"],
        "target_mask": batch["target_mask"],
    }
    out_cached = model.forward_cached(cached_batch)
    out_raw = model.forward(batch)
    assert torch.allclose(out_cached.loss, out_raw.loss, atol=1e-5)


def test_forward_cached_casts_fp16_states_to_connector_dtype():
    model = _build_model()
    batch = _batch()
    with torch.no_grad():
        raw_encoder_states = model.encoder(batch["semantic_codes"], batch["speech_attention_mask"])

    cached_batch = {
        "speech_states": raw_encoder_states.to(torch.float16),
        "speech_mask": batch["speech_attention_mask"],
        "prefix_ids": batch["prefix_ids"],
        "prefix_mask": batch["prefix_mask"],
        "target_ids": batch["target_ids"],
        "target_mask": batch["target_mask"],
    }
    out = model.forward_cached(cached_batch)
    assert torch.isfinite(out.loss)


def test_kv_cached_generation_matches_full_recomputation():
    model = _build_model()
    model.eval()
    batch = _batch()
    with torch.no_grad():
        states = model.encoder(batch["semantic_codes"], batch["speech_attention_mask"])
    one = {
        "speech_states": states[:1],
        "speech_mask": batch["speech_attention_mask"][:1],
        "prefix_ids": batch["prefix_ids"][:1],
        "prefix_mask": batch["prefix_mask"][:1],
    }
    cached = model.generate_cached(one, eos_token_id=-1, max_new_tokens=4, use_kv_cache=True)
    uncached = model.generate_cached(one, eos_token_id=-1, max_new_tokens=4, use_kv_cache=False)
    assert cached == uncached
