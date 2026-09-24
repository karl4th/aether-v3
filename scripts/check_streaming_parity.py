"""Compare offline and random-chunk streaming decoding on real held-out rows."""

from __future__ import annotations

import argparse
import json
import statistics
import time

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
    assert upsampler_state is not None
    tail, _, _ = model.flush_stream(encoder_state, upsampler_state)
    if tail is not None:
        pieces.append(tail)
    return torch.cat(pieces, dim=1)


def _long_stream_diagnostic(
    model: AetherCTCModel, device: torch.device, *, minutes: int = 10
) -> dict[str, float | int | bool]:
    frame_count = minutes * 60 * 25 // 2
    codes = torch.randint(
        0, model.encoder.cfg.semantic_vocab_size, (1, frame_count), device=device
    )
    encoder_state = None
    upsampler_state = None
    latencies_ms: list[float] = []
    finite = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_start = torch.cuda.memory_allocated(device)
    else:
        memory_start = 0
    with torch.no_grad():
        for start in range(0, frame_count, 32):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits, encoder_state, upsampler_state = model.forward_chunk(
                    codes[:, start : start + 32], encoder_state, upsampler_state
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            finite = finite and bool(torch.isfinite(logits).all())
    assert encoder_state is not None
    assert upsampler_state is not None
    encoder_cache_frames = max(cache.key.shape[2] for cache in encoder_state.layer_caches if cache)
    upsampler_cache_frames = max(
        cache.key.shape[2] for cache in upsampler_state.layer_caches if cache
    )
    memory_end = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "long_stream_minutes": minutes,
        "long_stream_frames": frame_count,
        "long_stream_outputs_finite": finite,
        "encoder_cache_frames": encoder_cache_frames,
        "upsampler_cache_frames": upsampler_cache_frames,
        "first_20_chunk_latency_ms": statistics.median(latencies_ms[:20]),
        "last_20_chunk_latency_ms": statistics.median(latencies_ms[-20:]),
        "allocated_memory_start_mib": memory_start / 1024**2,
        "allocated_memory_end_mib": memory_end / 1024**2,
        "peak_memory_mib": peak_memory / 1024**2,
    }


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

    report = {
        "checkpoint_step": snapshot.get("step"),
        "dtype": "float32",
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
    }
    report.update(_long_stream_diagnostic(model, device))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
