"""Stage 2B training loop for cached SLUE-SQA-5 question speech states."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from aether_v3.config import ExperimentConfig
from aether_v3.data.stage2_collate import collate_stage2_batch
from aether_v3.data.stage2_dataset import Stage2ShardDataset
from aether_v3.models.aether_speech_llm import AetherSpeechLLM
from aether_v3.training.scheduler import build_scheduler
from aether_v3.training.stage2_utils import (
    answer_exact_match,
    answer_f1,
    append_jsonl,
    save_stage2_checkpoint,
)

logger = logging.getLogger(__name__)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _trainable_parameters(model: torch.nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate_stage2(
    model: AetherSpeechLLM,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_examples: int,
    max_new_tokens: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    f1_scores: list[float] = []
    exact_scores: list[float] = []
    seen = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model.forward_cached(batch)
        if output.loss is not None:
            losses.append(float(output.loss))
        # Generation is deliberately per-example; see generate_cached.
        for index in range(batch["speech_states"].shape[0]):
            one = {
                key: value[index : index + 1] if torch.is_tensor(value) else value
                for key, value in batch.items()
                if key not in {"sample_ids", "references"}
            }
            ids = model.generate_cached(one, tokenizer.eos_token_id, max_new_tokens=max_new_tokens)[
                0
            ]
            prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
            references = raw_batch["references"][index]
            f1_scores.append(answer_f1(prediction, references))
            exact_scores.append(answer_exact_match(prediction, references))
            seen += 1
            if seen >= max_examples:
                break
        if seen >= max_examples:
            break
    model.train()
    return {
        "val_loss": sum(losses) / max(1, len(losses)),
        "answer_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "exact_match": sum(exact_scores) / max(1, len(exact_scores)),
    }


def run_stage2_training(
    config: ExperimentConfig,
    run_dir: str | Path,
    train_cache_dir: str | Path,
    validation_cache_dir: str | Path,
    model: AetherSpeechLLM | None = None,
) -> Path:
    """Train Connector on cached speech, writing every artifact directly to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "periodic").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.stage2_train.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.llm.model_id, revision=config.llm.revision)
    model = model or AetherSpeechLLM(
        config.aether_speech, config.connector, config.llm, speech_frozen=True
    )
    model.to(device)
    llm_dtype = next(model.llm.parameters()).dtype
    model.connector.to(dtype=llm_dtype)

    train_data = Stage2ShardDataset(train_cache_dir, shuffle=True, seed=config.stage2_train.seed)
    val_data = Stage2ShardDataset(validation_cache_dir, shuffle=False)
    train_loader = DataLoader(
        train_data,
        batch_size=config.stage2_train.batch_size,
        num_workers=config.stage2_train.num_workers,
        collate_fn=collate_stage2_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(val_data, batch_size=1, collate_fn=collate_stage2_batch)
    params = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        params, lr=config.stage2_train.lr, weight_decay=config.stage2_train.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        config.stage2_train.warmup_steps,
        config.stage2_train.max_steps,
        config.stage2_train.min_lr_ratio,
    )
    provenance = {
        "git_commit": _git_commit(),
        "llm_model_id": config.llm.model_id,
        "llm_revision": config.llm.revision,
        "stage1_repo_id": config.stage2_train.stage1_repo_id,
        "stage1_filename": config.stage2_train.stage1_filename,
        "dataset_id": config.stage2_data.dataset_id,
        "dataset_config": config.stage2_data.dataset_config,
    }
    config_dict = dataclasses.asdict(config)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    step = 0
    best = {"val_loss": float("inf"), "answer_f1": -1.0, "exact_match": -1.0}
    resume = config.stage2_train.resume_from
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        for name, value in checkpoint["trainable_model"].items():
            named[name].data.copy_(value.to(device=device, dtype=named[name].dtype))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        best.update(checkpoint.get("metrics", {}))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    running = 0.0
    started = time.time()
    while step < config.stage2_train.max_steps:
        for _ in range(config.stage2_train.grad_accum_steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                train_data.set_epoch(train_data.epoch + 1)
                iterator = iter(train_loader)
                raw_batch = next(iterator)
            batch = _to_device(raw_batch, device)
            output = model.forward_cached(batch)
            if output.loss is None:
                raise RuntimeError("Qwen returned no training loss")
            loss = output.loss / config.stage2_train.grad_accum_steps
            loss.backward()
            running += float(loss)
        torch.nn.utils.clip_grad_norm_(params, config.stage2_train.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % config.stage2_train.log_interval == 0:
            record = {
                "step": step,
                "train_loss": running / config.stage2_train.log_interval,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(run_dir / "log.jsonl", record)
            logger.info("%s", record)
            running = 0.0

        metrics: dict[str, float] | None = None
        if step % config.stage2_train.eval_interval == 0:
            metrics = evaluate_stage2(
                model,
                val_loader,
                tokenizer,
                device,
                config.stage2_train.eval_max_examples,
                config.stage2_train.generation_max_new_tokens,
            )
            append_jsonl(run_dir / "log.jsonl", {"step": step, **metrics})
            for metric, better in (
                ("val_loss", metrics["val_loss"] < best["val_loss"]),
                ("answer_f1", metrics["answer_f1"] > best["answer_f1"]),
                ("exact_match", metrics["exact_match"] > best["exact_match"]),
            ):
                if better:
                    best[metric] = metrics[metric]
                    save_stage2_checkpoint(
                        run_dir / f"best_{metric}.pt",
                        model,
                        optimizer,
                        scheduler,
                        step,
                        best,
                        config_dict,
                        provenance,
                    )
        if step % config.stage2_train.save_interval == 0:
            save_stage2_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                step,
                best,
                config_dict,
                provenance,
            )
            save_stage2_checkpoint(
                run_dir / "periodic" / f"step_{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                step,
                best,
                config_dict,
                provenance,
            )
    save_stage2_checkpoint(
        run_dir / "last.pt", model, optimizer, scheduler, step, best, config_dict, provenance
    )
    return run_dir
