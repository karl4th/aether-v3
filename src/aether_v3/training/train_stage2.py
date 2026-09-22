"""Stage 2B training loop for cached SLUE-SQA-5 question speech states."""

from __future__ import annotations

import dataclasses
import datetime as dt
import gc
import hashlib
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from aether_v3.config import ExperimentConfig
from aether_v3.data.stage2_collate import collate_stage2_batch
from aether_v3.data.stage2_dataset import Stage2ShardDataset
from aether_v3.eval.metrics import compute_cer, compute_wer
from aether_v3.models.aether_speech_llm import AetherSpeechLLM
from aether_v3.training.dist_utils import unwrap_model
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_trainable_weights(
    model: AetherSpeechLLM, checkpoint_path: str | Path, device: torch.device | str
) -> dict[str, Any]:
    """Load only trainable Stage 2 weights, leaving optimizer/schedule fresh."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("trainable_model")
    if not isinstance(source, dict):
        raise ValueError(f"Checkpoint has no trainable_model mapping: {path}")
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    supplied = set(source)
    missing = expected - supplied
    unexpected = supplied - set(named)
    if missing or unexpected:
        raise ValueError(
            f"Incompatible trainable weights: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name, value in source.items():
        named[name].data.copy_(value.to(device=device, dtype=named[name].dtype))
    return checkpoint


@torch.inference_mode()
def evaluate_stage2(
    model: AetherSpeechLLM,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_examples: int,
    max_new_tokens: int,
    task: str,
) -> dict[str, Any]:
    model.eval()
    cuda_memory_before: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_memory_before = {
            "eval_gpu_allocated_before_gb": torch.cuda.memory_allocated(device) / 2**30,
            "eval_gpu_reserved_before_gb": torch.cuda.memory_reserved(device) / 2**30,
        }
    loss_sum = 0.0
    loss_examples = 0
    f1_scores: list[float] = []
    exact_scores: list[float] = []
    predictions: list[str] = []
    primary_references: list[str] = []
    seen = 0
    progress = tqdm(total=max_examples, desc="Stage 2 eval", unit="utt", dynamic_ncols=True)
    for raw_batch in loader:
        remaining = max_examples - seen
        raw_batch = {
            key: value[:remaining] if torch.is_tensor(value) or isinstance(value, list) else value
            for key, value in raw_batch.items()
        }
        batch = _to_device(raw_batch, device)
        output = model.forward_cached(batch)
        if output.loss is not None:
            current_batch_size = int(batch["speech_states"].shape[0])
            loss_sum += float(output.loss) * current_batch_size
            loss_examples += current_batch_size
        generation_batch = {
            key: batch[key] for key in ("speech_states", "speech_mask", "prefix_ids", "prefix_mask")
        }
        generated = model.generate_cached(
            generation_batch, tokenizer.eos_token_id, max_new_tokens=max_new_tokens
        )
        for index, ids in enumerate(generated):
            prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
            references = raw_batch["references"][index]
            predictions.append(prediction)
            primary_references.append(references[0])
            f1_scores.append(answer_f1(prediction, references))
            exact_scores.append(answer_exact_match(prediction, references))
            seen += 1
            progress.update(1)
            if seen >= max_examples:
                break
        del output, batch, generation_batch, generated
        if seen >= max_examples:
            break
    progress.close()
    model.train()
    metrics: dict[str, Any] = {
        "val_loss": loss_sum / max(1, loss_examples),
        "examples": list(zip(primary_references[:5], predictions[:5], strict=True)),
    }
    if task == "transcription":
        metrics["wer"] = compute_wer(primary_references, predictions)
        metrics["cer"] = compute_cer(primary_references, predictions)
    else:
        metrics["answer_f1"] = sum(f1_scores) / max(1, len(f1_scores))
        metrics["exact_match"] = sum(exact_scores) / max(1, len(exact_scores))
    metrics.update(cuda_memory_before)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        metrics.update(
            {
                "eval_gpu_allocated_after_cleanup_gb": torch.cuda.memory_allocated(device) / 2**30,
                "eval_gpu_reserved_after_cleanup_gb": torch.cuda.memory_reserved(device) / 2**30,
            }
        )
    return metrics


def run_stage2_training(
    config: ExperimentConfig,
    run_dir: str | Path,
    train_cache_dir: str | Path,
    validation_cache_dir: str | Path,
    model: AetherSpeechLLM | None = None,
    tokenizer: Any | None = None,
) -> Path:
    """Train Connector on cached speech, writing every artifact directly to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "periodic").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.stage2_train.seed)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        config.llm.model_id, revision=config.llm.revision
    )
    model = model or AetherSpeechLLM(
        config.aether_speech, config.connector, config.llm, speech_frozen=True
    )
    # This loop consumes cached AetherSpeech states, so the frozen encoder is
    # never called. Keep it on CPU instead of wasting VRAM for the entire run.
    model.encoder.to("cpu")
    model.llm.to(device)  # type: ignore[arg-type]  # transformers stub mis-infers .to
    model.connector.to(device)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    llm_dtype = next(model.llm.parameters()).dtype
    model.connector.to(dtype=llm_dtype)

    init_checkpoint: dict[str, Any] | None = None
    init_path: Path | None = None
    if config.stage2_train.init_trainable_from:
        if config.stage2_train.resume_from:
            raise ValueError("Set only one of init_trainable_from and resume_from")
        init_path = Path(config.stage2_train.init_trainable_from)
        init_checkpoint = load_stage2_trainable_weights(model, init_path, device)
        logger.info(
            "initialized trainable weights from %s at source step %s; "
            "optimizer and scheduler are fresh",
            init_path,
            init_checkpoint.get("step", "unknown"),
        )

    train_data = Stage2ShardDataset(train_cache_dir, shuffle=True, seed=config.stage2_train.seed)
    val_data = Stage2ShardDataset(validation_cache_dir, shuffle=False)
    train_loader = DataLoader(
        train_data,
        batch_size=config.stage2_train.batch_size,
        num_workers=config.stage2_train.num_workers,
        collate_fn=collate_stage2_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.stage2_train.eval_batch_size,
        collate_fn=collate_stage2_batch,
    )
    params = _trainable_parameters(model)
    unwrapped_model = cast(AetherSpeechLLM, unwrap_model(model))
    optimizer = torch.optim.AdamW(
        params, lr=config.stage2_train.lr, weight_decay=config.stage2_train.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        config.stage2_train.warmup_steps,
        config.stage2_train.max_steps,
        config.stage2_train.min_lr_ratio,
    )
    provenance: dict[str, Any] = {
        "git_commit": _git_commit(),
        "llm_model_id": config.llm.model_id,
        "llm_revision": config.llm.revision,
        "stage1_repo_id": config.stage2_train.stage1_repo_id,
        "stage1_filename": config.stage2_train.stage1_filename,
        "dataset_id": config.stage2_data.dataset_id,
        "dataset_config": config.stage2_data.dataset_config,
    }
    if init_path is not None and init_checkpoint is not None:
        provenance["init_trainable_from"] = str(init_path)
        provenance["init_trainable_sha256"] = _sha256(init_path)
        provenance["init_trainable_source_step"] = init_checkpoint.get("step")
        provenance["init_trainable_source_provenance"] = init_checkpoint.get("provenance", {})
    config_dict = dataclasses.asdict(config)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    step = 0
    if config.stage2_data.task == "transcription":
        best = {"val_loss": float("inf"), "wer": float("inf"), "cer": float("inf")}
    else:
        best = {"val_loss": float("inf"), "answer_f1": -1.0, "exact_match": -1.0}
    resume = config.stage2_train.resume_from
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        for name, value in checkpoint["trainable_model"].items():
            named[name].data.copy_(value.to(device=device, dtype=named[name].dtype))
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        best.update(checkpoint.get("metrics", {}))

    previous_output_scale: float | None = None
    scale_warning_emitted = False

    def enforce_output_scale_guard(current_step: int) -> float:
        nonlocal previous_output_scale, scale_warning_emitted
        output_scale = float(unwrapped_model.connector.bridge.output_scale.detach())
        abort_max = config.stage2_train.output_scale_abort_max
        abort_delta = config.stage2_train.output_scale_abort_step_delta
        step_delta = (
            abs(output_scale - previous_output_scale) if previous_output_scale is not None else 0.0
        )
        safe = (
            math.isfinite(output_scale)
            and (abort_max is None or abs(output_scale) <= abort_max)
            and (abort_delta is None or step_delta <= abort_delta)
        )
        if safe:
            warn_max = config.stage2_train.output_scale_warn_max
            if warn_max is not None and abs(output_scale) > warn_max and not scale_warning_emitted:
                warning = {
                    "step": current_step,
                    "status": "warning",
                    "reason": "bridge_output_scale_warning",
                    "bridge_output_scale": output_scale,
                    "output_scale_warn_max": warn_max,
                }
                append_jsonl(run_dir / "log.jsonl", warning)
                logger.warning("%s", warning)
                scale_warning_emitted = True
            previous_output_scale = output_scale
            return output_scale
        reason = {
            "step": current_step,
            "status": "aborted",
            "reason": "bridge_output_scale_guard",
            "bridge_output_scale": output_scale,
            "output_scale_abort_max": abort_max,
            "bridge_output_scale_step_delta": step_delta,
            "output_scale_abort_step_delta": abort_delta,
        }
        append_jsonl(run_dir / "log.jsonl", reason)
        save_stage2_checkpoint(
            run_dir / "abort_output_scale.pt",
            model,
            optimizer,
            scheduler,
            current_step,
            best,
            config_dict,
            provenance,
        )
        logger.error("%s", reason)
        raise RuntimeError(
            "Bridge output scale guard stopped training at step "
            f"{current_step}: scale={output_scale}, limit={abort_max}. "
            f"Diagnostic checkpoint: {run_dir / 'abort_output_scale.pt'}"
        )

    enforce_output_scale_guard(step)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    running = 0.0
    started = time.time()

    plateau_metric = config.stage2_train.plateau_metric
    plateau_best: float | None = None
    plateau_bad_evals = 0
    plateau_history: list[dict[str, Any]] = []

    def run_evaluation(current_step: int) -> bool:
        nonlocal plateau_best, plateau_bad_evals
        metrics = evaluate_stage2(
            model,
            val_loader,
            tokenizer,
            device,
            config.stage2_train.eval_max_examples,
            config.stage2_train.generation_max_new_tokens,
            config.stage2_data.task,
        )
        examples = metrics.pop("examples")
        append_jsonl(run_dir / "log.jsonl", {"step": current_step, **metrics, "examples": examples})
        logger.info("eval step %d | %s", current_step, metrics)
        for reference, prediction in examples:
            logger.info("  ref: %r", reference)
            logger.info("  hyp: %r", prediction)
        directions = {
            "val_loss": "min",
            "wer": "min",
            "cer": "min",
            "answer_f1": "max",
            "exact_match": "max",
        }
        for metric, value in metrics.items():
            if metric not in best:
                continue
            better = value < best[metric] if directions[metric] == "min" else value > best[metric]
            if better:
                best[metric] = value
                save_stage2_checkpoint(
                    run_dir / f"best_{metric}.pt",
                    model,
                    optimizer,
                    scheduler,
                    current_step,
                    best,
                    config_dict,
                    provenance,
                )
                logger.info("saved best_%s.pt at step %d", metric, current_step)

        if not config.stage2_train.plateau_enabled:
            return False
        if plateau_metric not in metrics:
            raise ValueError(f"Plateau metric {plateau_metric!r} is absent from evaluation")
        plateau_start = (
            config.stage2_train.plateau_start_step
            if config.stage2_train.plateau_start_step is not None
            else config.stage2_train.warmup_steps
        )
        value = float(metrics[plateau_metric])
        entry = {"step": current_step, "value": value, "eligible": current_step >= plateau_start}
        plateau_history.append(entry)
        if current_step < plateau_start:
            plateau_best = value if plateau_best is None else min(plateau_best, value)
            entry["significant_best"] = plateau_best
            entry["consecutive_without_min_delta"] = 0
            return False
        if plateau_best is None or plateau_best - value >= config.stage2_train.plateau_min_delta:
            plateau_best = value
            plateau_bad_evals = 0
        else:
            plateau_bad_evals += 1
        entry["significant_best"] = plateau_best
        entry["consecutive_without_min_delta"] = plateau_bad_evals
        if plateau_bad_evals < config.stage2_train.plateau_patience_evals:
            return False

        report = {
            "status": "stopped",
            "reason": "validation_plateau",
            "step": current_step,
            "metric": plateau_metric,
            "min_delta": config.stage2_train.plateau_min_delta,
            "patience_evals": config.stage2_train.plateau_patience_evals,
            "plateau_start_step": plateau_start,
            "best_significant_value": plateau_best,
            "current_value": value,
            "history": plateau_history,
        }
        (run_dir / "plateau_report.json").write_text(json.dumps(report, indent=2))
        append_jsonl(run_dir / "log.jsonl", report)
        save_stage2_checkpoint(
            run_dir / "plateau_stop.pt",
            model,
            optimizer,
            scheduler,
            current_step,
            best,
            config_dict,
            provenance,
        )
        logger.warning("%s", report)
        return True

    if step == 0 and (0 in config.stage2_train.eval_steps or not config.stage2_train.eval_steps):
        logger.info("running mandatory eval@step0")
        run_evaluation(0)
    stopped_for_plateau = False
    start_step = step
    progress = tqdm(
        total=config.stage2_train.max_steps,
        initial=step,
        desc="Stage 2 train",
        unit="step",
        disable=not config.stage2_train.show_progress_bar,
        dynamic_ncols=True,
    )
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
            running += float(loss.detach())
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(params, config.stage2_train.grad_clip_norm)
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)

        output_scale = enforce_output_scale_guard(step)

        if step % config.stage2_train.log_interval == 0:
            elapsed = time.time() - started
            completed_this_run = step - start_step
            steps_per_second = completed_this_run / max(elapsed, 1e-9)
            remaining_steps = config.stage2_train.max_steps - step
            eta_seconds = remaining_steps / max(steps_per_second, 1e-9)
            record = {
                "step": step,
                "train_loss": running / config.stage2_train.log_interval,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_per_second,
                "remaining_steps": remaining_steps,
                "progress_percent": 100.0 * step / config.stage2_train.max_steps,
                "eta_seconds": eta_seconds,
                "estimated_completion_utc": (
                    dt.datetime.now(dt.UTC) + dt.timedelta(seconds=eta_seconds)
                ).isoformat(),
                "grad_norm": grad_norm,
                "bridge_output_scale": output_scale,
            }
            if device.type == "cuda":
                record["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
                record["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
                record["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
            append_jsonl(run_dir / "log.jsonl", record)
            logger.info("%s", record)
            progress.set_postfix(
                loss=f"{record['train_loss']:.4f}",
                scale=f"{output_scale:.4f}",
                remaining=remaining_steps,
            )
            running = 0.0

        if step in config.stage2_train.eval_steps or (
            not config.stage2_train.eval_steps and step % config.stage2_train.eval_interval == 0
        ):
            stopped_for_plateau = run_evaluation(step)
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
            logger.info("saved last.pt and periodic checkpoint at step %d", step)
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
        if stopped_for_plateau:
            break
    progress.close()
    save_stage2_checkpoint(
        run_dir / "last.pt", model, optimizer, scheduler, step, best, config_dict, provenance
    )
    completion = {
        "status": "stopped_early" if stopped_for_plateau else "completed",
        "reason": "validation_plateau" if stopped_for_plateau else "max_steps",
        "step": step,
        "max_steps": config.stage2_train.max_steps,
        "best": best,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(completion, indent=2))
    return run_dir
