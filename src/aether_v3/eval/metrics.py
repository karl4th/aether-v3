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
