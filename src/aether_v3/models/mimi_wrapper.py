"""Frozen Mimi encoder wrapper — extracts semantic codes only.

Verified against `transformers` (kyutai/mimi, transformers 5.17):
- `AutoFeatureExtractor.from_pretrained("kyutai/mimi")` resolves to an
  `EncodecFeatureExtractor` with `sampling_rate == 24000`. Called with a list
  of mono float32 arrays (`raw_audio=[...]`, `padding=True`) it returns
  `input_values` of shape (batch, 1, samples) and `padding_mask` of shape
  (batch, samples) (int32, 1=valid/0=padding) - *not* (batch, channels,
  samples) as the docstring for `decode()` suggests.
- `model.config.frame_rate == 12.5`, `model.config.codebook_size == 2048`,
  `model.config.num_semantic_quantizers == 1` -> codebook index 0 is the
  2048-way semantic codebook, matching the architecture spec exactly.
- `model.encode(input_values, padding_mask=padding_mask, num_quantizers=1)`
  returns a `MimiEncoderOutput` with `.audio_codes` of shape
  (batch, 1, frames); only the semantic codebook is computed (no wasted
  compute on the 31 acoustic codebooks we don't use).
- `model.get_audio_codes_mask(padding_mask)` returns the correct frame-level
  boolean mask (batch, frames), right-padded, derived from the model's own
  downsampling stride — this was confirmed against a real forward pass and
  is used instead of any hand-computed stride, since Mimi's raw-sample-per
  -frame ratio is not simply `sampling_rate / frame_rate` due to an internal
  causal-conv downsampling stage.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, MimiModel


class FrozenMimi:
    """Loads a frozen kyutai/mimi model and extracts semantic code sequences."""

    def __init__(
        self,
        pretrained_id: str = "kyutai/mimi",
        num_quantizers: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = MimiModel.from_pretrained(pretrained_id)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)  # type: ignore[arg-type]  # transformers' .to() stub mis-infers this overload
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_id)
        self.num_quantizers = num_quantizers
        self.device = torch.device(device)
        self.target_sample_rate: int = self.feature_extractor.sampling_rate

    @torch.no_grad()
    def encode_semantic(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        """Encode a batch of mono float32 waveforms into semantic code id sequences.

        Args:
            waveforms: list of 1D float32 arrays already at `target_sample_rate`
                (resampling is the caller's job - see `_RawAudioDataset` in
                `mimi_cache.py`, where it's done in parallel `DataLoader`
                workers instead of serially here, blocking the GPU).

        Returns:
            list of 1D int32 arrays (one per input), each containing the
            semantic codebook ids (values in [0, 2047]) for that utterance,
            with any batch padding already stripped.
        """
        if not waveforms:
            return []

        inputs = self.feature_extractor(
            raw_audio=waveforms,
            sampling_rate=self.target_sample_rate,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device)

        outputs = self.model.encode(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=self.num_quantizers,
            return_dict=True,
        )
        assert not isinstance(outputs, tuple)  # return_dict=True rules out the tuple return path
        assert outputs.audio_codes is not None  # always set when encode() is given input_values
        codes = outputs.audio_codes[:, 0, :]  # (batch, frames) - semantic codebook only
        frame_mask = self.model.get_audio_codes_mask(padding_mask)  # (batch, frames) bool
        lengths = frame_mask.sum(dim=-1).tolist()

        codes = codes.to("cpu")
        return [codes[i, : lengths[i]].numpy().astype(np.int32) for i in range(len(waveforms))]
