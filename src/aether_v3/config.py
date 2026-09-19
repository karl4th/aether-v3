"""Experiment configuration for AETHER STT phase 1 (CTC branch).

Loaded from YAML (see configs/ctc_base.yaml) into plain dataclasses. Kept
deliberately flat/non-nested-generic so it can be built with simple
``DataclassType(**raw_dict)`` calls without an extra dependency.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class MimiConfig:
    pretrained_id: str = "kyutai/mimi"
    sampling_rate: int = 24000
    num_quantizers: int = 1  # semantic codebook only (index 0)


@dataclasses.dataclass
class AetherSpeechConfig:
    semantic_vocab_size: int = 2048
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ffn_size: int = 3072
    dropout: float = 0.1
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096


@dataclasses.dataclass
class CTCConfig:
    vocab_size: int = 257  # 256 UTF-8 byte values + 1 blank
    blank_id: int = 256


@dataclasses.dataclass
class DataConfig:
    dataset_id: str = "openslr/librispeech_asr"
    fallback_dataset_id: str = "distil-whisper/librispeech_asr"
    # Each entry is "<hf_config_name>/<split_name>", e.g. "clean/train.100".
    train_splits: list[str] = dataclasses.field(
        default_factory=lambda: ["clean/train.100", "clean/train.360"]
    )
    validation_splits: list[str] = dataclasses.field(
        default_factory=lambda: ["clean/validation.clean", "other/validation.other"]
    )
    test_splits: list[str] = dataclasses.field(
        default_factory=lambda: ["clean/test.clean", "other/test.other"]
    )
    cache_dir: str = "data_cache/ctc_base"
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 20.0
    extraction_batch_size: int = 16


@dataclasses.dataclass
class TrainConfig:
    output_dir: str = "runs/ctc_base"
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 200_000
    warmup_steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    grad_clip_norm: float = 1.0
    eval_interval: int = 2000
    save_interval: int = 2000
    log_interval: int = 50
    num_workers: int = 4
    seed: int = 1337
    amp_dtype: str = "bfloat16"
    wandb_project: str | None = None
    resume_from: str | None = None


@dataclasses.dataclass
class ExperimentConfig:
    mimi: MimiConfig = dataclasses.field(default_factory=MimiConfig)
    aether_speech: AetherSpeechConfig = dataclasses.field(default_factory=AetherSpeechConfig)
    ctc: CTCConfig = dataclasses.field(default_factory=CTCConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig(
        mimi=MimiConfig(**raw.get("mimi", {})),
        aether_speech=AetherSpeechConfig(**raw.get("aether_speech", {})),
        ctc=CTCConfig(**raw.get("ctc", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    raw = {
        "mimi": dataclasses.asdict(config.mimi),
        "aether_speech": dataclasses.asdict(config.aether_speech),
        "ctc": dataclasses.asdict(config.ctc),
        "data": dataclasses.asdict(config.data),
        "train": dataclasses.asdict(config.train),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
