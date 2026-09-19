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
import torchaudio
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
        self.model.to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_id)
        self.num_quantizers = num_quantizers
        self.device = torch.device(device)
        self.target_sample_rate: int = self.feature_extractor.sampling_rate

    def _resample(self, waveform: np.ndarray, orig_sample_rate: int) -> np.ndarray:
        if orig_sample_rate == self.target_sample_rate:
            return np.asarray(waveform, dtype=np.float32)
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        resampled = torchaudio.functional.resample(tensor, orig_sample_rate, self.target_sample_rate)
        return resampled.numpy()

    @torch.no_grad()
    def encode_semantic(
        self, waveforms: list[np.ndarray], orig_sample_rate: int
    ) -> list[np.ndarray]:
        """Encode a batch of mono float32 waveforms into semantic code id sequences.

        Args:
            waveforms: list of 1D float32 arrays at `orig_sample_rate`.
            orig_sample_rate: sample rate of every array in `waveforms`.

        Returns:
            list of 1D int32 arrays (one per input), each containing the
            semantic codebook ids (values in [0, 2047]) for that utterance,
            with any batch padding already stripped.
        """
        if not waveforms:
            return []
        resampled = [self._resample(w, orig_sample_rate) for w in waveforms]

        inputs = self.feature_extractor(
            raw_audio=resampled,
            sampling_rate=self.target_sample_rate,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device)

        outputs = self.model.encode(
            input_values, padding_mask=padding_mask, num_quantizers=self.num_quantizers
        )
        codes = outputs.audio_codes[:, 0, :]  # (batch, frames) - semantic codebook only
        frame_mask = self.model.get_audio_codes_mask(padding_mask)  # (batch, frames) bool
        lengths = frame_mask.sum(dim=-1).tolist()

        codes = codes.to("cpu")
        return [codes[i, : lengths[i]].numpy().astype(np.int32) for i in range(len(waveforms))]
