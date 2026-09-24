"""Fault-tolerant Weights & Biases monitoring for one persistent run URL."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aether_v3.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class WandbMonitor:
    run: Any
    metadata_path: Path
    log_examples: bool
    failed: bool = False

    def log(self, payload: dict[str, Any], *, step: int) -> str | None:
        if self.failed:
            return None
        try:
            self.run.log({"trainer/global_step": step, **payload}, step=step)
        except Exception as exc:  # noqa: BLE001
            self.failed = True
            logger.exception("W&B logging failed; local training will continue")
            return f"{type(exc).__name__}: {exc}"
        return None

    def log_evaluation(self, metrics: dict[str, Any], *, step: int) -> str | None:
        payload: dict[str, Any] = {
            "eval/loss": float(metrics["loss"]),
            "eval/ctc_loss": float(metrics["ctc_loss"]),
            "eval/semantic_loss": float(metrics["semantic_loss"]),
            "eval/wer": float(metrics["wer"]),
            "eval/cer": float(metrics["cer"]),
            "eval/short_query_wer": float(metrics["short_query_wer"]),
            "eval/short_query_examples": float(metrics["short_query_examples"]),
            "eval/empty_hypothesis_rate": float(metrics["empty_hypothesis_rate"]),
            "eval/repetition_collapse_rate": float(metrics["repetition_collapse_rate"]),
            "eval/utterance_wer_gte_100_rate": float(metrics["utterance_wer_gte_100_rate"]),
            "eval/invalid_utf8_rate": float(metrics["invalid_utf8_rate"]),
            "eval/mean_length_ratio": float(metrics["mean_hypothesis_reference_length_ratio"]),
            "eval/truncated_hypothesis_rate": float(metrics["truncated_hypothesis_rate"]),
            "eval/catastrophic_failure_rate": float(metrics["catastrophic_failure_rate"]),
        }
        payload = {
            key: value
            for key, value in payload.items()
            if not isinstance(value, float) or math.isfinite(value)
        }
        if self.log_examples:
            try:
                import wandb

                payload["eval/examples"] = wandb.Table(
                    columns=["reference", "hypothesis"], data=metrics["examples"]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create W&B evaluation table: %s", exc)
        return self.log(payload, step=step)

    def finish(self, *, exit_code: int) -> None:
        try:
            self.run.finish(exit_code=exit_code)
        except Exception:  # noqa: BLE001
            logger.exception("W&B finish failed; local artifacts are unaffected")


def _write_monitoring_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def start_wandb_monitor(
    config: ExperimentConfig, run_path: Path, run_id: str
) -> WandbMonitor | None:
    project = config.train.wandb_project
    if not project:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        message = "WANDB_API_KEY is required when train.wandb_project is configured"
        if config.train.wandb_required:
            raise RuntimeError(message)
        logger.warning("%s; monitoring disabled", message)
        return None

    try:
        import wandb

        wandb_run = wandb.init(
            project=project,
            entity=config.train.wandb_entity,
            dir=str(run_path),
            group=config.train.wandb_group,
            tags=config.train.wandb_tags,
            id=run_id,
            name=run_id,
            resume="allow",
            config=asdict(config),
            allow_val_change=True,
        )
        if wandb_run is None:
            raise RuntimeError("wandb.init returned no run")
        wandb_run.define_metric("trainer/global_step")
        for namespace in ("train/*", "eval/*", "progress/*"):
            wandb_run.define_metric(namespace, step_metric="trainer/global_step")
        monitor = WandbMonitor(
            run=wandb_run,
            metadata_path=run_path / "monitoring.json",
            log_examples=config.train.wandb_log_examples,
        )
        _write_monitoring_metadata(
            monitor.metadata_path,
            {
                "provider": "wandb",
                "run_id": run_id,
                "run_url": str(wandb_run.url),
                "project_url": str(wandb_run.project_url),
                "project": project,
                "entity": wandb_run.entity,
            },
        )
        return monitor
    except Exception:  # noqa: BLE001
        if config.train.wandb_required:
            raise
        logger.exception("Could not initialize optional W&B monitoring")
        return None
