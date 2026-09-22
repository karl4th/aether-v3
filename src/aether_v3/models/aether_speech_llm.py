"""AetherSpeechLLM: Stage 2 model - AetherSpeech + AetherConnector + Qwen LLM.

Stage 2 spec Sec.3, 15-16, 47-48. Scope is deliberately limited to what
Phase 0-2 (docs/stage2_spec.md) actually needs: a frozen AetherSpeech
encoder feeding a trainable Connector into a frozen pretrained LLM, with
the LLM's own `labels=` loss (standard next-token cross-entropy, shifted
internally by `transformers`). LoRA (Phase 3), CTC auxiliary loss (Phase
4+, gated on AetherSpeech becoming trainable - see the comment in
`ctc_upsampler.py`), and text-teacher KD (Phase 6) are later, explicitly
gated phases and are not wired in here; adding them should not require
restructuring this class, just extending it once that phase is reached.

The LLM is never run under `torch.no_grad()`, even when frozen (Sec.33):
gradients must still flow from the loss back through Qwen's frozen graph
into `inputs_embeds`, so the Connector (trainable) receives gradients.
`requires_grad_(False)` on the LLM's parameters only stops *it* from
accumulating gradients, it does not cut the graph.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from aether_v3.config import AetherSpeechConfig, ConnectorConfig, LLMConfig
from aether_v3.models.aether_speech import AetherSpeechEncoder
from aether_v3.models.connector import AetherConnector


@dataclasses.dataclass
class BuiltInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor


@dataclasses.dataclass
class Stage2Output:
    loss: torch.Tensor | None
    logits: torch.Tensor


class AetherSpeechLLM(nn.Module):
    def __init__(
        self,
        speech_cfg: AetherSpeechConfig,
        connector_cfg: ConnectorConfig,
        llm_cfg: LLMConfig,
        speech_frozen: bool = True,
        llm: PreTrainedModel | None = None,
    ) -> None:
        """`llm`: inject an already-constructed causal LM (e.g. a tiny
        randomly-initialized `Qwen3ForCausalLM` in tests, or the Phase 0
        Qwen3-0.6B harness model) instead of downloading `llm_cfg.model_id`
        from the Hub - the default `None` loads the real pretrained model."""
        super().__init__()
        self.encoder = AetherSpeechEncoder(speech_cfg)
        self.connector = AetherConnector(speech_cfg.hidden_size, connector_cfg)
        self.llm = (
            llm
            if llm is not None
            else AutoModelForCausalLM.from_pretrained(
                llm_cfg.model_id, dtype=getattr(torch, llm_cfg.dtype)
            )
        )

        self.speech_frozen = speech_frozen
        self.llm_frozen = llm_cfg.frozen
        if self.speech_frozen:
            self.encoder.requires_grad_(False)
        if self.llm_frozen:
            self.llm.requires_grad_(False)
        if llm_cfg.gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        self.train(self.training)

    def train(self, mode: bool = True) -> AetherSpeechLLM:
        super().train(mode)
        # Frozen submodules always stay in eval mode (no dropout/etc.),
        # regardless of the wrapper's own train()/eval() calls.
        if self.speech_frozen:
            self.encoder.eval()
        if self.llm_frozen:
            self.llm.eval()
        return self

    def encode_speech(
        self, semantic_codes: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.speech_frozen:
            with torch.no_grad():
                hidden = self.encoder(semantic_codes, attention_mask)
        else:
            hidden = self.encoder(semantic_codes, attention_mask)
        return self.connector(hidden, attention_mask)

    def build_inputs_embeds(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        speech_embeds: torch.Tensor,
        speech_mask: torch.Tensor | None,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> BuiltInputs:
        """Assembles `[prefix] <SPEECH_START> [speech] <SPEECH_END> [target]`
        per example (Sec.15-16), right-padded to the batch's max total
        length. `target_ids`/`target_mask` should already include EOS if the
        model is meant to learn to emit it (Sec.16: "[Y1 ... YN EOS]")."""
        device = speech_embeds.device
        dtype = speech_embeds.dtype
        batch_size, hidden_size = speech_embeds.shape[0], speech_embeds.shape[-1]

        embed_tokens = self.llm.get_input_embeddings()
        prefix_embeds = embed_tokens(prefix_ids)
        target_embeds = embed_tokens(target_ids)

        prefix_lengths = prefix_mask.sum(dim=1)
        speech_lengths = (
            speech_mask.sum(dim=1)
            if speech_mask is not None
            else torch.full((batch_size,), speech_embeds.shape[1], device=device)
        )
        target_lengths = target_mask.sum(dim=1)
        total_lengths = prefix_lengths + 1 + speech_lengths + 1 + target_lengths
        max_len = int(total_lengths.max())

        inputs_embeds = torch.zeros(batch_size, max_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            p_len = int(prefix_lengths[i])
            s_len = int(speech_lengths[i])
            t_len = int(target_lengths[i])
            seq = torch.cat(
                [
                    prefix_embeds[i, :p_len],
                    self.connector.speech_start.unsqueeze(0).to(dtype),
                    speech_embeds[i, :s_len],
                    self.connector.speech_end.unsqueeze(0).to(dtype),
                    target_embeds[i, :t_len],
                ],
                dim=0,
            )
            seq_len = seq.shape[0]
            inputs_embeds[i, :seq_len] = seq
            attention_mask[i, :seq_len] = True
            target_start = p_len + 1 + s_len + 1
            labels[i, target_start : target_start + t_len] = target_ids[i, :t_len]

        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return BuiltInputs(inputs_embeds, attention_mask, labels, position_ids)

    def forward(self, batch: dict[str, torch.Tensor]) -> Stage2Output:
        """Raw-codes path: runs the (frozen) encoder itself. `batch` keys:
        semantic_codes, speech_attention_mask, prefix_ids, prefix_mask,
        target_ids, target_mask."""
        speech_embeds, speech_mask = self.encode_speech(
            batch["semantic_codes"], batch["speech_attention_mask"]
        )
        return self._forward_with_speech_embeds(
            speech_embeds,
            speech_mask,
            batch["prefix_ids"],
            batch["prefix_mask"],
            batch["target_ids"],
            batch["target_mask"],
        )

    def forward_cached(self, batch: dict[str, torch.Tensor]) -> Stage2Output:
        """Pre-computed-encoder-states path (Stage 2 data pipeline v0, see
        `aether_v3.data.stage2_cache`): skips `self.encoder` entirely
        (frozen anyway while cached states are in use - see
        docs/stage2_spec.md Sec.16, "invalidated the moment AetherSpeech
        becomes trainable") and runs the trainable Connector directly on
        the cached states. `batch` keys: speech_states, speech_mask (from
        `collate_stage2_batch`), plus prefix_ids/prefix_mask (added by the
        training loop, constant for the whole run - not this collator's
        job) and target_ids/target_mask."""
        connector_dtype = next(self.connector.parameters()).dtype
        speech_states = batch["speech_states"].to(connector_dtype)
        speech_embeds, speech_mask = self.connector(speech_states, batch["speech_mask"])
        return self._forward_with_speech_embeds(
            speech_embeds,
            speech_mask,
            batch["prefix_ids"],
            batch["prefix_mask"],
            batch["target_ids"],
            batch["target_mask"],
        )

    def _forward_with_speech_embeds(
        self,
        speech_embeds: torch.Tensor,
        speech_mask: torch.Tensor | None,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Stage2Output:
        built = self.build_inputs_embeds(
            prefix_ids, prefix_mask, speech_embeds, speech_mask, target_ids, target_mask
        )
        outputs = self.llm(
            inputs_embeds=built.inputs_embeds,
            attention_mask=built.attention_mask,
            position_ids=built.position_ids,
            labels=built.labels,
            use_cache=False,
        )
        return Stage2Output(loss=outputs.loss, logits=outputs.logits)

    @torch.no_grad()
    def generate_cached(
        self,
        batch: dict[str, torch.Tensor],
        eos_token_id: int,
        max_new_tokens: int = 64,
        use_kv_cache: bool = True,
    ) -> list[list[int]]:
        """Greedy generation for a right-padded batch of cached speech states."""
        batch_size = batch["speech_states"].shape[0]
        if not use_kv_cache and batch_size != 1:
            raise ValueError("Batched generation requires use_kv_cache=True")
        connector_dtype = next(self.connector.parameters()).dtype
        states = batch["speech_states"].to(connector_dtype)
        speech_embeds, speech_mask = self.connector(states, batch["speech_mask"])

        empty_ids = torch.empty(batch_size, 0, dtype=torch.long, device=states.device)
        empty_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=states.device)
        built = self.build_inputs_embeds(
            batch["prefix_ids"],
            batch["prefix_mask"],
            speech_embeds,
            speech_mask,
            empty_ids,
            empty_mask,
        )
        embeds = built.inputs_embeds
        attention_mask = built.attention_mask
        prefix_lengths = attention_mask.sum(dim=1)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        active = torch.ones(batch_size, dtype=torch.bool, device=states.device)
        past_key_values = None
        last_tokens: torch.Tensor | None = None
        for generation_step in range(max_new_tokens):
            position_ids = attention_mask.long().cumsum(-1) - 1
            if past_key_values is not None:
                assert last_tokens is not None
                out = self.llm(
                    input_ids=last_tokens.unsqueeze(1),
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                out = self.llm(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_kv_cache,
                )
            past_key_values = out.past_key_values if use_kv_cache else None
            if generation_step == 0:
                rows = torch.arange(batch_size, device=states.device)
                next_logits = out.logits[rows, prefix_lengths - 1]
            else:
                next_logits = out.logits[:, -1]
            next_tokens = next_logits.argmax(dim=-1)
            token_values = next_tokens.detach().cpu().tolist()
            active_values = active.detach().cpu().tolist()
            for index, (token, is_active) in enumerate(
                zip(token_values, active_values, strict=True)
            ):
                if is_active and token != eos_token_id:
                    generated[index].append(token)
            active = active & next_tokens.ne(eos_token_id)
            if not bool(active.any().item()):
                break
            if not use_kv_cache:
                token_embed = self.llm.get_input_embeddings()(next_tokens.unsqueeze(1)).to(
                    embeds.dtype
                )
                embeds = torch.cat([embeds, token_embed], dim=1)
            last_tokens = torch.where(
                active, next_tokens, torch.full_like(next_tokens, eos_token_id)
            )
            attention_mask = torch.cat([attention_mask, active.unsqueeze(1)], dim=1)
        return generated
