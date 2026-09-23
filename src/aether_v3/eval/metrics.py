"""WER/CER via jiwer."""

from __future__ import annotations

import jiwer


def _non_empty(strings: list[str]) -> list[str]:
    # jiwer chokes on all-empty references; a lone space is a safe stand-in.
    return [s if s.strip() else " " for s in strings]


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return float("nan")
    return jiwer.wer(_non_empty(references), _non_empty(hypotheses))


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return float("nan")
    return jiwer.cer(_non_empty(references), _non_empty(hypotheses))


def _has_repetition_collapse(text: str, run_length: int = 4) -> bool:
    words = text.upper().split()
    return any(
        len(set(words[start : start + run_length])) == 1
        for start in range(max(0, len(words) - run_length + 1))
    )


def compute_failure_metrics(
    references: list[str],
    hypotheses: list[str],
    audio_seconds: list[float] | None = None,
) -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if audio_seconds is not None and len(audio_seconds) != len(references):
        raise ValueError("audio_seconds must align with references")
    if not references:
        return {name: float("nan") for name in _FAILURE_METRIC_NAMES}

    empty = [not hypothesis.strip() for hypothesis in hypotheses]
    repetition = [_has_repetition_collapse(hypothesis) for hypothesis in hypotheses]
    invalid_utf8 = ["\ufffd" in hypothesis for hypothesis in hypotheses]
    utterance_wer = [
        compute_wer([reference], [hypothesis])
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    length_ratios = [
        len(hypothesis.split()) / max(1, len(reference.split()))
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    truncated = [ratio < 0.5 for ratio in length_ratios]
    catastrophic = [
        is_empty or is_repetition or is_invalid or is_truncated or sample_wer >= 1.0
        for is_empty, is_repetition, is_invalid, is_truncated, sample_wer in zip(
            empty, repetition, invalid_utf8, truncated, utterance_wer, strict=True
        )
    ]
    count = len(references)
    short_indices = (
        [index for index, seconds in enumerate(audio_seconds) if 1.0 <= seconds <= 5.0]
        if audio_seconds is not None
        else []
    )
    short_wer = (
        compute_wer(
            [references[index] for index in short_indices],
            [hypotheses[index] for index in short_indices],
        )
        if short_indices
        else float("nan")
    )
    return {
        "short_query_wer": short_wer,
        "short_query_examples": float(len(short_indices)),
        "empty_hypothesis_rate": sum(empty) / count,
        "repetition_collapse_rate": sum(repetition) / count,
        "utterance_wer_gte_100_rate": sum(value >= 1.0 for value in utterance_wer) / count,
        "invalid_utf8_rate": sum(invalid_utf8) / count,
        "mean_hypothesis_reference_length_ratio": sum(length_ratios) / count,
        "truncated_hypothesis_rate": sum(truncated) / count,
        "catastrophic_failure_rate": sum(catastrophic) / count,
    }


_FAILURE_METRIC_NAMES = (
    "short_query_wer",
    "short_query_examples",
    "empty_hypothesis_rate",
    "repetition_collapse_rate",
    "utterance_wer_gte_100_rate",
    "invalid_utf8_rate",
    "mean_hypothesis_reference_length_ratio",
    "truncated_hypothesis_rate",
    "catastrophic_failure_rate",
)
