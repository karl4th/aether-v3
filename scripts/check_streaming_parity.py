"""Compare offline and random-chunk streaming decoding on real held-out rows."""

from __future__ import annotations

import argparse
import json

import torch

from aether_v3.config import load_config
from aether_v3.data.loquacious import load_loquacious_split
from aether_v3.data.tokenizer import byte_ids_to_text
from aether_v3.eval.decode import greedy_ctc_decode
from aether_v3.eval.metrics import compute_cer, compute_wer
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.training.checkpoint import load_model_weights


def _stream(model: AetherCTCModel, codes: torch.Tensor) -> torch.Tensor:
    sizes = (1, 2, 3, 5, 7, 16, 32)
    encoder_state = None
    upsampler_state = None
    pieces = []
    start = 0
    index = 0
    while start < codes.shape[1]:
        size = sizes[index % len(sizes)]
        logits, encoder_state, upsampler_state = model.forward_chunk(
            codes[:, start : start + size], encoder_state, upsampler_state
        )
        pieces.append(logits)
        start += size
        index += 1
    assert encoder_state is not None
    tail, _, _ = model.flush_stream(encoder_state, upsampler_state)
    if tail is not None:
        pieces.append(tail)
    return torch.cat(pieces, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_loquacious_split(
        config.data, config.data.validation_split, max_samples=args.samples
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AetherCTCModel(config.aether_speech, config.ctc, config.semantic_prediction).to(device)
    snapshot = load_model_weights(args.checkpoint, model, map_location=device)
    model.eval()

    refs: list[str] = []
    offline_hyps: list[str] = []
    streaming_hyps: list[str] = []
    durations: list[float] = []
    max_logit_difference = 0.0
    with torch.no_grad():
        for sample_index in range(len(dataset)):
            codes, target, duration = dataset[sample_index]
            codes = codes.unsqueeze(0).to(device)
            mask = torch.ones_like(codes, dtype=torch.bool)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                offline = model(codes, mask)
                streaming = _stream(model, codes)
            max_logit_difference = max(
                max_logit_difference,
                float((offline.float() - streaming.float()).abs().max().item()),
            )
            output_length = torch.tensor([offline.shape[1]])
            offline_hyps.extend(greedy_ctc_decode(offline.float().cpu(), output_length))
            streaming_hyps.extend(greedy_ctc_decode(streaming.float().cpu(), output_length))
            refs.append(byte_ids_to_text(target.tolist()))
            durations.append(duration)

    short = [index for index, duration in enumerate(durations) if duration <= 4.0]
    long = [index for index, duration in enumerate(durations) if duration >= 12.0]

    def subset_wer(indices: list[int], hypotheses: list[str]) -> float | None:
        if not indices:
            return None
        return compute_wer(
            [refs[index] for index in indices], [hypotheses[index] for index in indices]
        )

    print(
        json.dumps(
            {
                "checkpoint_step": snapshot.get("step"),
                "samples": len(refs),
                "hypothesis_mismatches": sum(
                    left != right for left, right in zip(offline_hyps, streaming_hyps, strict=True)
                ),
                "max_abs_logit_difference": max_logit_difference,
                "offline_wer": compute_wer(refs, offline_hyps),
                "streaming_wer": compute_wer(refs, streaming_hyps),
                "offline_cer": compute_cer(refs, offline_hyps),
                "streaming_cer": compute_cer(refs, streaming_hyps),
                "short_samples": len(short),
                "short_offline_wer": subset_wer(short, offline_hyps),
                "short_streaming_wer": subset_wer(short, streaming_hyps),
                "long_samples": len(long),
                "long_offline_wer": subset_wer(long, offline_hyps),
                "long_streaming_wer": subset_wer(long, streaming_hyps),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
