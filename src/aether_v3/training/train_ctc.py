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
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from aether_v3.config import ExperimentConfig, load_config, save_config
from aether_v3.data.cached_dataset import CTCCachedDataset
from aether_v3.data.collate import collate_ctc_batch
from aether_v3.data.tokenizer import byte_ids_to_text
from aether_v3.eval.decode import greedy_ctc_decode
from aether_v3.eval.metrics import compute_cer, compute_wer
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.models.ctc_head import compute_ctc_loss
from aether_v3.training.checkpoint import load_checkpoint, save_checkpoint
from aether_v3.training.dist_utils import (
    get_rank,
    is_distributed,
    is_main_process,
    setup_distributed,
    teardown_distributed,
    wrap_model,
)
from aether_v3.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


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


def build_dataloader(
    dataset: CTCCachedDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    distributed: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler: DistributedSampler | None = None
    use_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        use_shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_ctc_batch,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader, sampler


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
) -> dict:
    """Evaluates on every example in `loader` - no batch cap.

    A fixed-prefix cap (e.g. "first N batches") would evaluate the same
    non-random subset every time (this loader is never shuffled), silently
    biasing both the reported WER/CER and which checkpoint gets saved as
    "best". The configured validation splits are a few thousand utterances
    total, cheap enough to score in full each time.
    """
    model.eval()
    all_refs: list[str] = []
    all_hyps: list[str] = []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _amp_autocast(device, amp_dtype):
            log_probs = model(batch["semantic_codes"], batch["attention_mask"])
            loss = compute_ctc_loss(
                log_probs,
                batch["targets"],
                batch["input_lengths"],
                batch["target_lengths"],
                blank_id,
            )
        total_loss += loss.item()
        n_batches += 1
        hyps = greedy_ctc_decode(log_probs.float().cpu(), batch["input_lengths"].cpu(), blank_id)
        refs = targets_to_texts(batch["targets"].cpu(), batch["target_lengths"].cpu())
        all_hyps.extend(hyps)
        all_refs.extend(refs)
    model.train()
    return {
        "loss": total_loss / max(1, n_batches),
        "wer": compute_wer(all_refs, all_hyps),
        "cer": compute_cer(all_refs, all_hyps),
        "examples": list(zip(all_refs[:5], all_hyps[:5], strict=True)),
    }


def run_training(config: ExperimentConfig) -> None:
    device = setup_distributed()
    distributed = is_distributed()
    torch.manual_seed(config.train.seed + get_rank())

    output_dir = Path(config.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    cache_dir = Path(config.data.cache_dir)
    train_ds = CTCCachedDataset(cache_dir / "train")
    val_ds = CTCCachedDataset(cache_dir / "validation")

    train_loader, train_sampler = build_dataloader(
        train_ds,
        config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        distributed=distributed,
        drop_last=True,
    )
    val_loader, _ = build_dataloader(
        val_ds,
        config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        distributed=False,
        drop_last=False,
    )

    model: torch.nn.Module = AetherCTCModel(config.aether_speech, config.ctc)
    model = wrap_model(model, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, config.train.warmup_steps, config.train.max_steps, config.train.min_lr_ratio
    )

    step = 0
    best_cer = float("inf")
    if config.train.resume_from:
        checkpoint = load_checkpoint(
            config.train.resume_from, model, optimizer, scheduler, map_location=device
        )
        step = checkpoint.get("step", 0)
        best_cer = checkpoint.get("best_cer", float("inf"))

    amp_dtype = getattr(torch, config.train.amp_dtype)
    blank_id = config.ctc.blank_id

    json_logger = JsonlLogger(output_dir / "log.jsonl") if is_main_process() else None
    wandb_run = None
    if is_main_process() and config.train.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.train.wandb_project,
                config={
                    "mimi": dataclasses.asdict(config.mimi),
                    "aether_speech": dataclasses.asdict(config.aether_speech),
                    "ctc": dataclasses.asdict(config.ctc),
                    "data": dataclasses.asdict(config.data),
                    "train": dataclasses.asdict(config.train),
                },
            )
        except ImportError:
            logger.warning("wandb_project is set but wandb is not installed; skipping.")

    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    epoch = 0
    running_loss = 0.0
    t0 = time.time()

    while step < config.train.max_steps:
        for micro_step in range(config.train.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)

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
                    log_probs = model(batch["semantic_codes"], batch["attention_mask"])
                    loss = compute_ctc_loss(
                        log_probs,
                        batch["targets"],
                        batch["input_lengths"],
                        batch["target_lengths"],
                        blank_id,
                    )
                    loss = loss / config.train.grad_accum_steps
                loss.backward()
            running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if is_main_process() and step % config.train.log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / config.train.log_interval
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step %d | loss %.4f | lr %.2e | %.2fs/%d-step",
                step,
                avg_loss,
                lr,
                elapsed,
                config.train.log_interval,
            )
            if json_logger is not None:
                json_logger.log(step=step, loss=avg_loss, lr=lr)
            if wandb_run:
                wandb_run.log({"train/loss": avg_loss, "train/lr": lr}, step=step)
            running_loss = 0.0
            t0 = time.time()

        if step % config.train.eval_interval == 0:
            metrics = evaluate(model, val_loader, device, amp_dtype, blank_id)
            if is_main_process():
                logger.info(
                    "eval @ step %d | loss %.4f | wer %.4f | cer %.4f",
                    step,
                    metrics["loss"],
                    metrics["wer"],
                    metrics["cer"],
                )
                for ref, hyp in metrics["examples"]:
                    logger.info("  ref: %r", ref)
                    logger.info("  hyp: %r", hyp)
                if json_logger is not None:
                    json_logger.log(
                        step=step,
                        eval_loss=metrics["loss"],
                        eval_wer=metrics["wer"],
                        eval_cer=metrics["cer"],
                    )
                if wandb_run:
                    wandb_run.log(
                        {
                            "eval/loss": metrics["loss"],
                            "eval/wer": metrics["wer"],
                            "eval/cer": metrics["cer"],
                        },
                        step=step,
                    )
                if metrics["cer"] < best_cer:
                    best_cer = metrics["cer"]
                    save_checkpoint(
                        output_dir / "best.pt", model, optimizer, scheduler, step, best_cer
                    )

        if is_main_process() and step % config.train.save_interval == 0:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, step, best_cer)

    if is_main_process():
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, step, best_cer)
        if json_logger is not None:
            json_logger.close()
    teardown_distributed()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
