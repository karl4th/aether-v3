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
    # Bounded left context for online inference. Mimi q0 emits at ~12.5 Hz,
    # so 256 frames retain roughly 20 seconds of history while keeping KV
    # memory and per-chunk attention cost independent of conversation length.
    streaming_left_context_frames: int = 256


@dataclasses.dataclass
class CTCConfig:
    vocab_size: int = 257  # 256 UTF-8 byte values + 1 blank
    blank_id: int = 256
    # CTC-only temporal upsampler: expands AetherSpeechEncoder's 12.5Hz
    # states to 12.5 * upsample_factor Hz before the CTC head (see
    # aether_v3.models.ctc_upsampler.CTCUpsampler). AetherSpeechEncoder's
    # own output is untouched - only the CTC branch runs at the higher
    # rate. Exists because CTC needs at least one input frame per target
    # label (plus separators for adjacent repeats - see
    # _ctc_min_input_length in aether_v3.data.mimi_cache), and 12.5
    # frames/sec is below typical English byte-rate (~12-18 bytes/sec),
    # making a large fraction of real utterances structurally
    # CTC-infeasible at the raw Mimi frame rate.
    upsample_factor: int = 4
    upsampler_num_layers: int = 2
    upsampler_num_heads: int = 12
    upsampler_ffn_size: int = 3072
    upsampler_dropout: float = 0.1
    upsampler_rope_theta: float = 10000.0
    # Bounded acoustic lookahead at Mimi's native 12.5 Hz rate. Five
    # frames add about 400 ms of algorithmic latency; zero is strict causal.
    lookahead_frames: int = 0


@dataclasses.dataclass
class SemanticPredictionConfig:
    """Auxiliary causal objective that prevents CTC-only representations."""

    enabled: bool = True
    weight: float = 0.25
    future_horizons: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4])


@dataclasses.dataclass
class DataConfig:
    backend: str = "hf_parquet"
    dataset_id: str = "manifestro/stage1_aether"
    dataset_revision: str | None = "4bb733b62abd021c4a153ff5196682912933588e"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    # Legacy LibriSpeech extraction inputs. They remain available for the
    # local_arrow backend and cache-building tools, but are not used by the
    # published LoquaciousSet semantic cache.
    fallback_dataset_id: str = "distil-whisper/librispeech_asr"
    # Each entry is "<hf_config_name>/<split_name>", e.g. "clean/train.100".
    train_splits: list[str] = dataclasses.field(
        default_factory=lambda: ["clean/train.100", "clean/train.360"]
    )
    # clean-only deliberately - see the comment in configs/ctc_base.yaml on
    # why "other/validation"/"other/test" are costly to include.
    validation_splits: list[str] = dataclasses.field(default_factory=lambda: ["clean/validation"])
    test_splits: list[str] = dataclasses.field(default_factory=lambda: ["clean/test"])
    cache_dir: str = "data_cache/ctc_base"
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 20.0
    extraction_batch_size: int = 32
    # Parallel CPU workers decoding/resampling audio during extraction, so
    # decode overlaps with Mimi's GPU encode instead of blocking it.
    extraction_num_workers: int = 4
    # Dynamic batches are capped by both examples and total unpadded Mimi
    # frames. At q0's 12.5 Hz, 16,384 frames are about 21.8 audio-minutes.
    max_semantic_frames_per_batch: int = 16_384
    length_bucket_size: int = 512


@dataclasses.dataclass
class TrainConfig:
    runs_dir: str = "runs"
    run_name: str = "stage1"
    run_id: str | None = None
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
    wandb_entity: str | None = None
    wandb_group: str = "stage1"
    wandb_tags: list[str] = dataclasses.field(default_factory=lambda: ["stage1", "streaming"])
    wandb_required: bool = False
    wandb_log_examples: bool = True
    # Fresh initialization imports model weights only. Resume restores the
    # complete state and continues inside the original run directory.
    init_encoder_from: str | None = None
    resume_run_from: str | None = None
    encoder_lr_multiplier: float = 1.0
    use_fused_adamw: bool = True
    show_progress: bool = True


@dataclasses.dataclass
class ArtifactConfig:
    # Authentication is intentionally environment-only. HF_TOKEN is used for
    # Hugging Face; Google Drive uses Application Default Credentials.
    google_drive_folder_id: str | None = None
    hf_model_repo_id: str | None = None
    hf_private: bool = True
    sync_on_interrupt: bool = True
    sync_on_completion: bool = True
    hf_publish_selections: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ExperimentConfig:
    mimi: MimiConfig = dataclasses.field(default_factory=MimiConfig)
    aether_speech: AetherSpeechConfig = dataclasses.field(default_factory=AetherSpeechConfig)
    ctc: CTCConfig = dataclasses.field(default_factory=CTCConfig)
    semantic_prediction: SemanticPredictionConfig = dataclasses.field(
        default_factory=SemanticPredictionConfig
    )
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    artifacts: ArtifactConfig = dataclasses.field(default_factory=ArtifactConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig(
        mimi=MimiConfig(**raw.get("mimi", {})),
        aether_speech=AetherSpeechConfig(**raw.get("aether_speech", {})),
        ctc=CTCConfig(**raw.get("ctc", {})),
        semantic_prediction=SemanticPredictionConfig(**raw.get("semantic_prediction", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        artifacts=ArtifactConfig(**raw.get("artifacts", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    raw = {
        "mimi": dataclasses.asdict(config.mimi),
        "aether_speech": dataclasses.asdict(config.aether_speech),
        "ctc": dataclasses.asdict(config.ctc),
        "semantic_prediction": dataclasses.asdict(config.semantic_prediction),
        "data": dataclasses.asdict(config.data),
        "train": dataclasses.asdict(config.train),
        "artifacts": dataclasses.asdict(config.artifacts),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
