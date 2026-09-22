"""Shared Stage 2 setup, checkpoint, prompt, and evaluation helpers."""

from __future__ import annotations

import json
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aether_v3.training.dist_utils import unwrap_model


def create_run_dir(drive_root: str | Path, now: datetime | None = None) -> Path:
    """Create an immutable timestamped run directory on Google Drive."""
    stamp = (now or datetime.now()).strftime("run%y%m%d-%H%M%S")
    path = Path(drive_root) / stamp
    path.mkdir(parents=True, exist_ok=False)
    (path / "periodic").mkdir()
    return path


def load_stage1_encoder(checkpoint_path: str | Path, encoder: nn.Module) -> dict[str, Any]:
    """Load only ``AetherCTCModel.encoder`` weights from a Stage 1 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        # Also accept a checkpoint containing the encoder state dict directly.
        encoder_keys = set(encoder.state_dict())
        if set(state).issubset(encoder_keys):
            encoder_state = state
    if not encoder_state:
        raise RuntimeError("Stage 1 checkpoint contains no AetherSpeech encoder weights")
    encoder.load_state_dict(encoder_state, strict=True)
    return checkpoint


def qa_prefix(document: str) -> str:
    return (
        "Answer the spoken question using only the document. Give a short direct answer.\n\n"
        f"Document:\n{document.strip()}\n\nSpoken question:\n"
    )


def transcription_prefix() -> str:
    return "Transcribe the following speech exactly. Output only the transcript:\n"


def answer_strings(answer_spans: Any) -> list[str]:
    """Normalize both HF Sequence layouts used by different datasets versions."""
    if isinstance(answer_spans, dict):
        values = answer_spans.get("answer", [])
    else:
        values = [span.get("answer", "") for span in answer_spans]
    return [str(answer).strip() for answer in values if str(answer).strip()]


def first_answer(answer_spans: Any) -> str:
    answers = answer_strings(answer_spans)
    if answers:
        return answers[0]
    raise ValueError("SLUE-SQA-5 row has no non-empty answer span")


def _normalize_answer(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def answer_exact_match(prediction: str, references: list[str]) -> float:
    pred = _normalize_answer(prediction)
    return float(any(pred == _normalize_answer(ref) for ref in references))


def answer_f1(prediction: str, references: list[str]) -> float:
    pred = _normalize_answer(prediction)
    best = 0.0
    for reference in references:
        ref = _normalize_answer(reference)
        if not pred or not ref:
            score = float(pred == ref)
        else:
            common = sum(min(pred.count(token), ref.count(token)) for token in set(pred))
            if common == 0:
                score = 0.0
            else:
                precision = common / len(pred)
                recall = common / len(ref)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def save_stage2_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Save trainable weights only; frozen Qwen and Stage 1 weights stay referenced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = unwrap_model(model)
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "trainable_model": trainable,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "metrics": metrics,
        "config": config,
        "provenance": provenance,
    }
    torch.save(payload, path)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with open(path, "a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
