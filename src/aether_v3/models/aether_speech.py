"""Streaming AetherSpeech encoder over Mimi semantic tokens.

The encoder is causal by construction: a state at frame ``t`` never depends on
future semantic frames. ``forward_chunk`` carries a bounded per-layer KV cache,
so online inference does not recompute the conversation prefix and memory does
not grow with conversation duration. At Mimi q0's ~12.5 Hz frame rate, one
semantic frame represents roughly 80 ms of audio.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether_v3.config import AetherSpeechConfig
from aether_v3.models.rope import apply_rope, build_rope_cache


class TransformerBlockConfig(Protocol):
    hidden_size: int
    num_heads: int
    ffn_size: int
    dropout: float


@dataclasses.dataclass(frozen=True)
class AttentionCache:
    """Projected keys/values and validity mask retained by one layer."""

    key: torch.Tensor
    value: torch.Tensor
    key_mask: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AetherSpeechStreamingState:
    """Explicit stream state; safe to keep separately for concurrent sessions."""

    layer_caches: tuple[AttentionCache | None, ...]
    frames_seen: int = 0


class RopeSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        *,
        causal: bool = False,
        cache: AttentionCache | None = None,
        max_cache_frames: int | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, AttentionCache | None]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        new_key = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        new_value = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        new_key = apply_rope(new_key, cos, sin)

        current_mask = (
            key_padding_mask
            if key_padding_mask is not None
            else torch.ones(b, t, dtype=torch.bool, device=x.device)
        )
        if cache is None:
            key = new_key
            value = new_value
            full_key_mask = current_mask
            cached_length = 0
        else:
            if cache.key.shape[0] != b:
                raise ValueError(
                    f"stream batch changed from {cache.key.shape[0]} to {b}; reset stream state"
                )
            key = torch.cat((cache.key, new_key), dim=2)
            value = torch.cat((cache.value, new_value), dim=2)
            full_key_mask = torch.cat((cache.key_mask, current_mask), dim=1)
            cached_length = cache.key.shape[2]

        attn_bias = torch.zeros(b, 1, t, key.shape[2], dtype=q.dtype, device=q.device)
        attn_bias.masked_fill_(~full_key_mask[:, None, None, :], float("-inf"))
        if causal:
            query_index = torch.arange(t, device=x.device)[:, None] + cached_length
            key_index = torch.arange(key.shape[2], device=x.device)[None, :]
            causal_mask = key_index <= query_index
            if max_cache_frames is not None:
                causal_mask &= key_index > query_index - max_cache_frames
            attn_bias.masked_fill_(~causal_mask[None, None, :, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)

        next_cache = None
        if return_cache:
            keep = key.shape[2] if max_cache_frames is None else min(key.shape[2], max_cache_frames)
            next_cache = AttentionCache(
                key=key[:, :, -keep:, :].detach(),
                value=value[:, :, -keep:, :].detach(),
                key_mask=full_key_mask[:, -keep:].detach(),
            )
        return self.o_proj(out), next_cache


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = RopeSelfAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.ffn = FeedForward(cfg.hidden_size, cfg.ffn_size, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        *,
        causal: bool = False,
        cache: AttentionCache | None = None,
        max_cache_frames: int | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, AttentionCache | None]:
        attn_out, next_cache = self.attn(
            self.norm1(x),
            cos,
            sin,
            key_padding_mask,
            causal=causal,
            cache=cache,
            max_cache_frames=max_cache_frames,
            return_cache=return_cache,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, next_cache


class AetherSpeechEncoder(nn.Module):
    def __init__(self, cfg: AetherSpeechConfig) -> None:
        super().__init__()
        if cfg.streaming_left_context_frames <= 0:
            raise ValueError("streaming_left_context_frames must be positive")
        self.cfg = cfg
        self.semantic_embedding = nn.Embedding(cfg.semantic_vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.head_dim = cfg.hidden_size // cfg.num_heads

    def init_streaming_state(self) -> AetherSpeechStreamingState:
        """Return an empty explicit state for a new independent audio stream."""
        return AetherSpeechStreamingState(layer_caches=(None,) * len(self.blocks))

    def reset_stream(self) -> AetherSpeechStreamingState:
        """Return empty state when an utterance or session boundary is observed."""
        return self.init_streaming_state()

    def flush_stream(
        self, state: AetherSpeechStreamingState
    ) -> tuple[torch.Tensor | None, AetherSpeechStreamingState]:
        """Finish a stream and reset its state.

        Strictly causal AetherSpeech buffers no future-dependent frames, so
        there is nothing delayed to emit.
        """
        return None, self.reset_stream()

    def forward_chunk(
        self,
        semantic_codes: torch.Tensor,
        state: AetherSpeechStreamingState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AetherSpeechStreamingState]:
        """Encode newly arrived semantic frames and return their states plus KV cache."""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "streaming batches cannot contain padded frames; keep one state per live stream"
            )
        return self._encode_chunk(semantic_codes, state, attention_mask, return_cache=True)

    def _encode_chunk(
        self,
        semantic_codes: torch.Tensor,
        state: AetherSpeechStreamingState | None,
        attention_mask: torch.Tensor | None,
        *,
        return_cache: bool,
    ) -> tuple[torch.Tensor, AetherSpeechStreamingState]:
        if semantic_codes.ndim != 2:
            raise ValueError("semantic_codes must have shape (batch, time)")
        if semantic_codes.shape[1] == 0:
            raise ValueError("streaming chunks must contain at least one frame")
        if attention_mask is not None and attention_mask.shape != semantic_codes.shape:
            raise ValueError("attention_mask must match semantic_codes shape")

        state = state or self.init_streaming_state()
        if len(state.layer_caches) != len(self.blocks):
            raise ValueError("stream state does not match encoder layer count")

        _, t = semantic_codes.shape
        x = self.semantic_embedding(semantic_codes)
        cos, sin = build_rope_cache(
            t,
            self.head_dim,
            self.cfg.rope_theta,
            x.device,
            x.dtype,
            position_offset=state.frames_seen,
        )
        next_caches: list[AttentionCache] = []
        for block, cache in zip(self.blocks, state.layer_caches, strict=True):
            x, next_cache = block(
                x,
                cos,
                sin,
                attention_mask,
                causal=True,
                cache=cache,
                max_cache_frames=self.cfg.streaming_left_context_frames,
                return_cache=return_cache,
            )
            if return_cache:
                assert next_cache is not None
                next_caches.append(next_cache)

        output = self.final_norm(x)
        next_state = AetherSpeechStreamingState(
            layer_caches=tuple(next_caches), frames_seen=state.frames_seen + t
        )
        return output, next_state

    def forward(
        self, semantic_codes: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Causal full-sequence path used for training and offline evaluation."""
        output, _ = self._encode_chunk(semantic_codes, None, attention_mask, return_cache=False)
        return output
