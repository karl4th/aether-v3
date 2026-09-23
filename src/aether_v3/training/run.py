"""Immutable run directories, provenance, status, and metric selections."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from aether_v3.config import ExperimentConfig, save_config


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _new_run_id(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("r%y%m%d-%H%M%S")
    revision = (_git_value("rev-parse", "--short=7", "HEAD") or "nogit")[:7]
    return f"{name}-{timestamp}-{revision}"


@dataclasses.dataclass
class RunDirectory:
    path: Path
    run_id: str

    @classmethod
    def create(cls, config: ExperimentConfig) -> RunDirectory:
        run_id = config.train.run_id or _new_run_id(config.train.run_name)
        path = Path(config.train.runs_dir) / run_id
        path.mkdir(parents=True, exist_ok=False)
        for child in ("checkpoints", "selections", "evaluations", "predictions", "profiles"):
            (path / child).mkdir()
        save_config(config, path / "config.resolved.yaml")

        diff = _git_value("diff", "--binary") or ""
        provenance = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
        }
        _atomic_json(path / "provenance.json", provenance)
        run = cls(path=path, run_id=run_id)
        run.write_status(state="created", step=0)
        return run

    @classmethod
    def resume(cls, path: str | Path) -> RunDirectory:
        run_path = Path(path)
        provenance = json.loads((run_path / "provenance.json").read_text())
        return cls(path=run_path, run_id=provenance["run_id"])

    def write_status(self, *, state: str, step: int, **details: Any) -> None:
        _atomic_json(
            self.path / "status.json",
            {
                "run_id": self.run_id,
                "state": state,
                "step": step,
                "updated_at": datetime.now(UTC).isoformat(),
                **details,
            },
        )


class MetricSelections:
    """Track independent best checkpoints without duplicating model files."""

    DIRECTIONS = {
        "eval_loss": "min",
        "eval_wer": "min",
        "eval_cer": "min",
        "macro_domain_wer": "min",
        "short_query_wer": "min",
        "streaming_wer": "min",
        "catastrophic_failure_rate": "min",
        "repetition_collapse_rate": "min",
        "empty_hypothesis_rate": "min",
    }

    def __init__(self, run: RunDirectory, best: dict[str, float] | None = None) -> None:
        self.run = run
        self.best = dict(best or {})

    def improvements(self, metrics: dict[str, float]) -> list[str]:
        improved: list[str] = []
        for metric, direction in self.DIRECTIONS.items():
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            if not math.isfinite(value):
                continue
            previous = self.best.get(metric)
            if previous is None or (value < previous if direction == "min" else value > previous):
                improved.append(metric)
        return improved

    def update(self, metrics: dict[str, float], step: int, checkpoint: Path) -> list[str]:
        improved = self.improvements(metrics)
        for metric in improved:
            direction = self.DIRECTIONS[metric]
            value = float(metrics[metric])
            self.best[metric] = value
            _atomic_json(
                self.run.path / "selections" / f"best_{metric.removeprefix('eval_')}.json",
                {
                    "metric": metric,
                    "direction": direction,
                    "value": value,
                    "step": step,
                    "checkpoint": os.path.relpath(checkpoint, self.run.path / "selections"),
                },
            )
        return improved
