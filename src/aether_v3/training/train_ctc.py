"""Phase 1 training entrypoint: the CTC branch, trained on precomputed Mimi caches.

Runs correctly single-process on 1 GPU (`python -m aether_v3.training.train_ctc
--config configs/ctc_base.yaml`) or under `torchrun` on N GPUs
(`torchrun --nproc_per_node=N -m aether_v3.training.train_ctc --config ...`)
with no code changes — distributed-ness is detected from `torchrun`'s
environment variables (see `dist_utils.py`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm.auto import tqdm

from aether_v3.config import ExperimentConfig, load_config
from aether_v3.data.cached_dataset import CTCCachedDataset
from aether_v3.data.collate import collate_ctc_batch
from aether_v3.data.frame_batch_sampler import FrameBudgetBatchSampler
from aether_v3.data.loquacious import dataset_lengths, load_loquacious_split
from aether_v3.data.tokenizer import byte_ids_to_text
from aether_v3.eval.decode import greedy_ctc_decode
from aether_v3.eval.metrics import compute_cer, compute_failure_metrics, compute_wer
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.models.ctc_head import compute_ctc_loss
from aether_v3.training.artifacts import (
    publish_checkpoint_to_huggingface,
    sync_run_backups,
)
from aether_v3.training.checkpoint import (
    load_encoder_weights,
    load_training_checkpoint,
    save_model_weights,
    save_training_checkpoint,
)
from aether_v3.training.dist_utils import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
    teardown_distributed,
    unwrap_model,
    wrap_model,
)
from aether_v3.training.optimizer import build_optimizer
from aether_v3.training.run import MetricSelections, RunDirectory
from aether_v3.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)

# Always log these absolute step numbers, regardless of `log_interval` - so
# the very first log line lands quickly (proof the run isn't hung) instead
# of only ever appearing every `log_interval` steps, which can look
# indistinguishable from a frozen process on a slow first few steps (CUDA
# kernel compilation, etc).
_EARLY_LOG_STEPS = frozenset({1, 2, 5, 10, 20})


def _amp_autocast(device: torch.device, dtype: torch.dtype):
    """Autocast context, or a no-op on non-accelerator devices.

    Measured empirically: bf16 autocast on an Apple Silicon CPU backend ran
    ~100-1000x slower per step than plain fp32 (most CPU kernels have no
    optimized bf16 path and fall back to a slow reference implementation).
    Autocast is a GPU tensor-core optimization; forcing it on CPU is a
    pessimization, not "extra safety margin", so it's skipped there.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a")

    def log(self, **kwargs) -> None:
        kwargs["time"] = time.time()
        self._f.write(json.dumps(kwargs) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class StopRequest:
    """Signal-safe flag checked at optimizer-step boundaries."""

    def __init__(self) -> None:
        self.signal_name: str | None = None

    def request(self, signum: int, _frame) -> None:
        self.signal_name = signal.Signals(signum).name


def _is_finite_across_ranks(value: torch.Tensor) -> bool:
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if is_distributed():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    distributed: bool,
    drop_last: bool,
    seed: int = 0,
    max_semantic_frames: int | None = None,
    bucket_size: int = 512,
) -> tuple[DataLoader, Sampler | None]:
    lengths = dataset_lengths(dataset)
    if max_semantic_frames is not None and lengths is not None:
        batch_sampler = FrameBudgetBatchSampler(
            lengths,
            max_frames=max_semantic_frames,
            max_examples=batch_size,
            shuffle=shuffle,
            seed=seed,
            bucket_size=bucket_size,
            rank=get_rank() if distributed else 0,
            world_size=get_world_size() if distributed else 1,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_ctc_batch,
            pin_memory=True,
        )
        return loader, batch_sampler

    sampler: DistributedSampler | None = None
    use_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        use_shuffle = False
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_ctc_batch,
        pin_memory=True,
        drop_last=drop_last,
        generator=generator,
    )
    return loader, sampler


def _iter_epoch(
    loader: DataLoader,
    sampler: Sampler | None,
    *,
    seed: int,
    epoch: int,
    skip_batches: int = 0,
):
    set_epoch = getattr(sampler, "set_epoch", None)
    if set_epoch is not None:
        set_epoch(epoch)
    if loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    iterator = iter(loader)
    for _ in range(skip_batches):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("checkpoint data position exceeds the epoch length") from exc
    return iterator


def targets_to_texts(targets: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    texts = []
    offset = 0
    for length in lengths.tolist():
        texts.append(byte_ids_to_text(targets[offset : offset + length].tolist()))
        offset += length
    return texts


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    blank_id: int,
    show_progress: bool = False,
) -> dict:
    """Evaluates on every example in `loader` - no batch cap.

    A fixed-prefix cap (e.g. "first N batches") would evaluate the same
    non-random subset every time (this loader is never shuffled), silently
    biasing both the reported WER/CER and which checkpoint gets saved as
    "best". The configured validation splits are a few thousand utterances
    total, cheap enough to score in full each time.
    """
    model.eval()
    # input_lengths are computed at Mimi's raw 12.5Hz rate (see
    # collate_ctc_batch); the model's CTC branch runs at
    # upsample_factor-times that (see CTCUpsampler), so lengths must be
    # scaled up to match before they're used for CTC loss/decoding.
    upsample_factor = unwrap_model(model).upsample_factor
    all_refs: list[str] = []
    all_hyps: list[str] = []
    all_audio_seconds: list[float] = []
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_semantic_loss = 0.0
    n_batches = 0
    batches = tqdm(
        loader,
        total=len(loader),
        desc="validation",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in batches:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        up_input_lengths = batch["input_lengths"] * upsample_factor
        with _amp_autocast(device, amp_dtype):
            outputs = model(
                batch["semantic_codes"],
                batch["attention_mask"],
                compute_semantic_loss=True,
            )
            assert isinstance(outputs, tuple)
            log_probs, semantic_loss = outputs
            ctc_loss = compute_ctc_loss(
                log_probs,
                batch["targets"],
                up_input_lengths,
                batch["target_lengths"],
                blank_id,
            )
            loss = ctc_loss + unwrap_model(model).semantic_prediction_weight * semantic_loss
        total_loss += loss.item()
        total_ctc_loss += ctc_loss.item()
        total_semantic_loss += semantic_loss.item()
        n_batches += 1
        hyps = greedy_ctc_decode(log_probs.float().cpu(), up_input_lengths.cpu(), blank_id)
        refs = targets_to_texts(batch["targets"].cpu(), batch["target_lengths"].cpu())
        all_hyps.extend(hyps)
        all_refs.extend(refs)
        if "audio_seconds" in batch:
            all_audio_seconds.extend(batch["audio_seconds"].float().cpu().tolist())
    model.train()
    metrics = {
        "loss": total_loss / max(1, n_batches),
        "ctc_loss": total_ctc_loss / max(1, n_batches),
        "semantic_loss": total_semantic_loss / max(1, n_batches),
        "wer": compute_wer(all_refs, all_hyps),
        "cer": compute_cer(all_refs, all_hyps),
        "examples": list(zip(all_refs[:5], all_hyps[:5], strict=True)),
    }
    metrics.update(
        compute_failure_metrics(
            all_refs,
            all_hyps,
            all_audio_seconds if len(all_audio_seconds) == len(all_refs) else None,
        )
    )
    return metrics


def run_training(config: ExperimentConfig) -> Path:
    if config.train.init_encoder_from and config.train.resume_run_from:
        raise ValueError("init_encoder_from and resume_run_from are mutually exclusive")

    device = setup_distributed()
    distributed = is_distributed()
    torch.manual_seed(config.train.seed + get_rank())

    run_path: str | None = None
    if is_main_process():
        run = (
            RunDirectory.resume(config.train.resume_run_from)
            if config.train.resume_run_from
            else RunDirectory.create(config)
        )
        run_path = str(run.path)
    if distributed:
        shared_path = [run_path]
        dist.broadcast_object_list(shared_path, src=0)
        run_path = shared_path[0]
    assert run_path is not None
    run = RunDirectory.resume(run_path)

    train_ds: Dataset
    val_ds: Dataset
    if config.data.backend == "hf_parquet":
        train_ds = load_loquacious_split(config.data, config.data.train_split)
        val_ds = load_loquacious_split(config.data, config.data.validation_split)
    elif config.data.backend == "local_arrow":
        cache_dir = Path(config.data.cache_dir)
        train_ds = CTCCachedDataset(cache_dir / "train")
        val_ds = CTCCachedDataset(cache_dir / "validation")
    else:
        raise ValueError(f"unsupported data backend: {config.data.backend!r}")
    train_loader, train_sampler = build_dataloader(
        train_ds,
        config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        distributed=distributed,
        drop_last=True,
        seed=config.train.seed,
        max_semantic_frames=config.data.max_semantic_frames_per_batch,
        bucket_size=config.data.length_bucket_size,
    )
    val_loader, _ = build_dataloader(
        val_ds,
        config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        distributed=False,
        drop_last=False,
        seed=config.train.seed,
        max_semantic_frames=config.data.max_semantic_frames_per_batch,
        bucket_size=config.data.length_bucket_size,
    )

    raw_model = AetherCTCModel(config.aether_speech, config.ctc, config.semantic_prediction)
    if config.train.init_encoder_from:
        load_encoder_weights(config.train.init_encoder_from, raw_model.encoder)
    ctc_upsample_factor = raw_model.upsample_factor
    model: torch.nn.Module = wrap_model(raw_model, device)
    optimizer = build_optimizer(model, config.train, device)
    scheduler = build_scheduler(
        optimizer, config.train.warmup_steps, config.train.max_steps, config.train.min_lr_ratio
    )

    step = 0
    epoch = 0
    batches_in_epoch = 0
    best_metrics: dict[str, float] = {}
    if config.train.resume_run_from:
        checkpoint = load_training_checkpoint(
            run.path / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            map_location=device,
        )
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        batches_in_epoch = checkpoint["batches_in_epoch"]
        best_metrics = checkpoint["best_metrics"]

    selections = MetricSelections(run, best_metrics)
    amp_dtype = getattr(torch, config.train.amp_dtype)
    blank_id = config.ctc.blank_id
    json_logger = JsonlLogger(run.path / "train.jsonl") if is_main_process() else None
    event_logger = JsonlLogger(run.path / "events.jsonl") if is_main_process() else None
    wandb_run = None
    if is_main_process() and config.train.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.train.wandb_project,
                name=run.run_id,
                config=dataclasses.asdict(config),
            )
        except ImportError:
            logger.warning("wandb_project is set but wandb is not installed; skipping.")

    stop_request = StopRequest()
    previous_handlers = {
        sig: signal.signal(sig, stop_request.request) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    progress = tqdm(
        total=config.train.max_steps,
        initial=step,
        desc=run.run_id,
        disable=not (is_main_process() and config.train.show_progress),
        dynamic_ncols=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = _iter_epoch(
        train_loader,
        train_sampler,
        seed=config.train.seed,
        epoch=epoch,
        skip_batches=batches_in_epoch,
    )
    running_loss = 0.0
    running_ctc_loss = 0.0
    running_semantic_loss = 0.0
    running_examples = 0
    running_semantic_frames = 0
    steps_since_log = 0
    t0 = time.time()
    termination_reason: str | None = None

    if is_main_process():
        run.write_status(state="running", step=step)
        assert event_logger is not None
        event_logger.log(event="run_started", run_id=run.run_id, step=step)

    try:
        while step < config.train.max_steps and stop_request.signal_name is None:
            step_loss = torch.zeros((), device=device)
            step_ctc_loss = torch.zeros((), device=device)
            step_semantic_loss = torch.zeros((), device=device)
            step_examples = 0
            step_semantic_frames = 0
            for micro_step in range(config.train.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    batches_in_epoch = 0
                    data_iter = _iter_epoch(
                        train_loader, train_sampler, seed=config.train.seed, epoch=epoch
                    )
                    batch = next(data_iter)
                batches_in_epoch += 1

                step_examples += int(batch["input_lengths"].numel())
                step_semantic_frames += int(batch["input_lengths"].sum().item())

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                is_last_micro_step = micro_step == config.train.grad_accum_steps - 1
                sync_ctx: contextlib.AbstractContextManager
                if not distributed or is_last_micro_step:
                    sync_ctx = contextlib.nullcontext()
                else:
                    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    sync_ctx = model.no_sync()
                with sync_ctx:
                    with _amp_autocast(device, amp_dtype):
                        outputs = model(
                            batch["semantic_codes"],
                            batch["attention_mask"],
                            compute_semantic_loss=True,
                        )
                        assert isinstance(outputs, tuple)
                        log_probs, semantic_loss = outputs
                        ctc_loss = compute_ctc_loss(
                            log_probs,
                            batch["targets"],
                            batch["input_lengths"] * ctc_upsample_factor,
                            batch["target_lengths"],
                            blank_id,
                        )
                        loss = (
                            ctc_loss
                            + unwrap_model(model).semantic_prediction_weight * semantic_loss
                        )
                        loss = loss / config.train.grad_accum_steps
                    if not _is_finite_across_ranks(loss):
                        termination_reason = "non_finite_loss"
                        break
                    loss.backward()
                    step_loss += loss.detach()
                    step_ctc_loss += ctc_loss.detach() / config.train.grad_accum_steps
                    step_semantic_loss += semantic_loss.detach() / config.train.grad_accum_steps
            if termination_reason:
                break

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip_norm
            )
            if not _is_finite_across_ranks(grad_norm):
                termination_reason = "non_finite_gradient"
                break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            steps_since_log += 1
            running_loss += float(step_loss.item())
            running_ctc_loss += float(step_ctc_loss.item())
            running_semantic_loss += float(step_semantic_loss.item())
            running_examples += step_examples
            running_semantic_frames += step_semantic_frames

            should_log = step in _EARLY_LOG_STEPS or steps_since_log >= config.train.log_interval
            if is_main_process() and should_log:
                elapsed = time.time() - t0
                avg_loss = running_loss / steps_since_log
                avg_ctc_loss = running_ctc_loss / steps_since_log
                avg_semantic_loss = running_semantic_loss / steps_since_log
                lr = scheduler.get_last_lr()[0]
                steps_per_sec = steps_since_log / elapsed if elapsed > 0 else 0.0
                examples_per_sec = running_examples * get_world_size() / elapsed
                # q0 is 12.5 Hz. This is numerically audio-hours processed
                # per wall-clock hour because both numerator and denominator
                # are converted from seconds by the same factor of 3600.
                audio_hours_per_hour = running_semantic_frames / 12.5 / elapsed * get_world_size()
                eta_seconds = (
                    (config.train.max_steps - step) / steps_per_sec
                    if steps_per_sec > 0
                    else float("inf")
                )
                vram_allocated = (
                    torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                )
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ctc=f"{avg_ctc_loss:.4f}",
                    semantic=f"{avg_semantic_loss:.4f}",
                    grad=f"{float(grad_norm):.3f}",
                    lr=f"{lr:.2e}",
                    eta=_format_duration(eta_seconds),
                )
                assert json_logger is not None
                json_logger.log(
                    event="train",
                    step=step,
                    epoch=epoch,
                    loss=avg_loss,
                    ctc_loss=avg_ctc_loss,
                    semantic_loss=avg_semantic_loss,
                    grad_norm=float(grad_norm),
                    lr=lr,
                    steps_per_second=steps_per_sec,
                    examples_per_second=examples_per_sec,
                    audio_hours_per_hour=audio_hours_per_hour,
                    vram_allocated_gib=vram_allocated,
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/ctc_loss": avg_ctc_loss,
                            "train/semantic_loss": avg_semantic_loss,
                            "train/grad_norm": float(grad_norm),
                            "train/lr": lr,
                            "train/steps_per_second": steps_per_sec,
                            "train/examples_per_second": examples_per_sec,
                            "train/audio_hours_per_hour": audio_hours_per_hour,
                            "system/vram_allocated_gib": vram_allocated,
                        },
                        step=step,
                    )
                running_loss = 0.0
                running_ctc_loss = 0.0
                running_semantic_loss = 0.0
                running_examples = 0
                running_semantic_frames = 0
                steps_since_log = 0
                t0 = time.time()

            if step % config.train.eval_interval == 0:
                eval_t0 = time.time()
                metrics = (
                    evaluate(
                        unwrap_model(model),
                        val_loader,
                        device,
                        amp_dtype,
                        blank_id,
                        show_progress=config.train.show_progress,
                    )
                    if is_main_process()
                    else None
                )
                if distributed:
                    dist.barrier()
                t0 += time.time() - eval_t0
                if is_main_process():
                    assert metrics is not None
                    metric_values = {
                        "eval_loss": float(metrics["loss"]),
                        "eval_ctc_loss": float(metrics["ctc_loss"]),
                        "eval_semantic_loss": float(metrics["semantic_loss"]),
                        "eval_wer": float(metrics["wer"]),
                        "eval_cer": float(metrics["cer"]),
                        "short_query_wer": float(metrics["short_query_wer"]),
                        "catastrophic_failure_rate": float(metrics["catastrophic_failure_rate"]),
                        "repetition_collapse_rate": float(metrics["repetition_collapse_rate"]),
                        "empty_hypothesis_rate": float(metrics["empty_hypothesis_rate"]),
                    }
                    evaluation_path = run.path / "evaluations" / f"validation_step_{step:08d}.json"
                    evaluation_path.write_text(
                        json.dumps({"step": step, **metrics}, indent=2) + "\n"
                    )
                    improved = selections.improvements(metric_values)
                    snapshot = run.path / "checkpoints" / f"step_{step:08d}_model.pt"
                    if improved:
                        save_model_weights(snapshot, model, step=step)
                        selections.update(metric_values, step, snapshot)
                    assert json_logger is not None
                    json_logger.log(step=step, event="validation", **metric_values)
                    if wandb_run:
                        wandb_run.log(metric_values, step=step)
                    if improved and event_logger is not None:
                        event_logger.log(event="new_best", step=step, metrics=improved)
                    if config.artifacts.hf_model_repo_id:
                        for metric in improved:
                            selection = metric.removeprefix("eval_")
                            if selection in config.artifacts.hf_publish_selections:
                                try:
                                    publish_checkpoint_to_huggingface(
                                        run.path,
                                        snapshot,
                                        config.artifacts.hf_model_repo_id,
                                        private=config.artifacts.hf_private,
                                        selection=selection,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    if event_logger is not None:
                                        event_logger.log(
                                            event="huggingface_publish_failed",
                                            selection=selection,
                                            error=str(exc),
                                        )

            if is_main_process() and step % config.train.save_interval == 0:
                save_training_checkpoint(
                    run.path / "checkpoints" / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    epoch=epoch,
                    batches_in_epoch=batches_in_epoch,
                    best_metrics=selections.best,
                )
                run.write_status(state="running", step=step, epoch=epoch)

        if stop_request.signal_name:
            termination_reason = f"signal_{stop_request.signal_name}"
    except BaseException as exc:
        termination_reason = f"exception_{type(exc).__name__}"
        raise
    finally:
        progress.close()
        if is_main_process():
            state = "completed" if step >= config.train.max_steps else "interrupted"
            if termination_reason and termination_reason.startswith("non_finite"):
                state = "failed"
            save_training_checkpoint(
                run.path / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                step=step,
                epoch=epoch,
                batches_in_epoch=batches_in_epoch,
                best_metrics=selections.best,
                termination_reason=termination_reason,
            )
            run.write_status(
                state=state,
                step=step,
                epoch=epoch,
                termination_reason=termination_reason,
            )
            should_sync = (state == "completed" and config.artifacts.sync_on_completion) or (
                state != "completed" and config.artifacts.sync_on_interrupt
            )
            if should_sync:
                errors = sync_run_backups(run.path, config.artifacts)
                if errors and event_logger is not None:
                    event_logger.log(event="artifact_sync_failed", errors=errors)
            if event_logger is not None:
                event_logger.log(event="run_finished", state=state, step=step)
                event_logger.close()
            if json_logger is not None:
                json_logger.close()
            if wandb_run:
                wandb_run.finish(exit_code=0 if state == "completed" else 1)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        teardown_distributed()

    if termination_reason and termination_reason.startswith("non_finite"):
        raise FloatingPointError(termination_reason)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
