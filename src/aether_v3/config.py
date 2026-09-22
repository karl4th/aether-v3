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


@dataclasses.dataclass
class ResamplerConfig:
    """AetherResampler: learned temporal compression of AetherSpeech's
    12.5Hz output (Stage 2 spec Sec.7-11). `ratio` must be a power of two;
    `ratio=1` (or `enabled=False`) makes the resampler an identity
    pass-through - this is the "R1" native-rate experiment arm, `ratio=4`
    is "R4". Kept as its own dataclass (rather than reusing CTCConfig's
    upsampler fields) since this branch feeds the LLM bridge, not CTC, and
    downsamples instead of upsamples.
    """

    enabled: bool = True
    ratio: int = 4
    num_heads: int = 12
    ffn_size: int = 3072
    dropout: float = 0.1
    conv_kernel: int = 5


@dataclasses.dataclass
class BridgeConfig:
    """AetherBridge: projects AetherSpeech/Resampler states into the frozen
    LLM's embedding space (Stage 2 spec Sec.12-13). `output_dim` must match
    the target LLM's hidden size (2560 for Qwen3-4B).
    """

    input_dim: int = 768
    intermediate_dim: int = 3072
    output_dim: int = 2560
    dropout: float = 0.1
    # Learned scalar multiplier applied after the final RMSNorm, so the
    # Bridge's output RMS can be tuned to match the LLM's text-embedding
    # RMS instead of assuming they start compatible (Stage 2 spec Sec.13).
    init_output_scale: float = 1.0


@dataclasses.dataclass
class ConnectorConfig:
    resampler: ResamplerConfig = dataclasses.field(default_factory=ResamplerConfig)
    bridge: BridgeConfig = dataclasses.field(default_factory=BridgeConfig)


@dataclasses.dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen3-4B"
    revision: str = "main"
    frozen: bool = True
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


@dataclasses.dataclass
class Stage2LossConfig:
    """Stage 2 spec Sec.16-17, 30-31, 37-40. `ctc_weight`/`kd_weight` are
    0.0 until the corresponding phase (AetherSpeech unfrozen / distillation
    experiment) is actually reached - see the phase gates in
    docs/stage2_spec.md.
    """

    lm_weight: float = 1.0
    ctc_weight: float = 0.0
    kd_weight: float = 0.0
    kd_direction: str = "teacher_to_student"
    kd_temperature: float = 2.0


@dataclasses.dataclass
class Stage2DataConfig:
    dataset_id: str = "asapp/slue-phase-2"
    dataset_config: str = "sqa5"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    task: str = "transcription"
    max_audio_seconds: float = 30.0
    cache_dir: str = "data_cache/stage2"


@dataclasses.dataclass
class Stage2TrainConfig:
    drive_root: str = "/content/drive/MyDrive/aether-v3/stage2"
    batch_size: int = 1
    grad_accum_steps: int = 16
    max_steps: int = 5_000
    warmup_steps: int = 200
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    show_progress_bar: bool = True
    eval_interval: int = 250
    eval_steps: list[int] = dataclasses.field(
        default_factory=lambda: [0, 500, 1000, 2000, 3000, 5000]
    )
    save_interval: int = 250
    eval_max_examples: int = 64
    eval_batch_size: int = 1
    generation_max_new_tokens: int = 64
    num_workers: int = 2
    seed: int = 1337
    amp_dtype: str = "bfloat16"
    stage1_repo_id: str = "manifestro/aetherASR-EN-v0.1"
    stage1_filename: str = "last.pt"
    stage1_revision: str = "main"
    resume_from: str | None = None
    # Weights-only initialization for a new experiment. Unlike resume_from,
    # this deliberately starts a fresh optimizer, scheduler, and step counter.
    init_trainable_from: str | None = None
    output_scale_abort_max: float | None = None
    output_scale_warn_max: float | None = None
    output_scale_abort_step_delta: float | None = None
    plateau_enabled: bool = False
    plateau_metric: str = "wer"
    plateau_min_delta: float = 0.005
    plateau_patience_evals: int = 3
    plateau_start_step: int | None = None


@dataclasses.dataclass
class DataConfig:
    dataset_id: str = "openslr/librispeech_asr"
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
    connector: ConnectorConfig = dataclasses.field(default_factory=ConnectorConfig)
    llm: LLMConfig = dataclasses.field(default_factory=LLMConfig)
    stage2_loss: Stage2LossConfig = dataclasses.field(default_factory=Stage2LossConfig)
    stage2_data: Stage2DataConfig = dataclasses.field(default_factory=Stage2DataConfig)
    stage2_train: Stage2TrainConfig = dataclasses.field(default_factory=Stage2TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig(
        mimi=MimiConfig(**raw.get("mimi", {})),
        aether_speech=AetherSpeechConfig(**raw.get("aether_speech", {})),
        ctc=CTCConfig(**raw.get("ctc", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        connector=ConnectorConfig(
            resampler=ResamplerConfig(**raw.get("connector", {}).get("resampler", {})),
            bridge=BridgeConfig(**raw.get("connector", {}).get("bridge", {})),
        ),
        llm=LLMConfig(**raw.get("llm", {})),
        stage2_loss=Stage2LossConfig(**raw.get("stage2_loss", {})),
        stage2_data=Stage2DataConfig(**raw.get("stage2_data", {})),
        stage2_train=Stage2TrainConfig(**raw.get("stage2_train", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    raw = {
        "mimi": dataclasses.asdict(config.mimi),
        "aether_speech": dataclasses.asdict(config.aether_speech),
        "ctc": dataclasses.asdict(config.ctc),
        "data": dataclasses.asdict(config.data),
        "train": dataclasses.asdict(config.train),
        "connector": dataclasses.asdict(config.connector),
        "llm": dataclasses.asdict(config.llm),
        "stage2_loss": dataclasses.asdict(config.stage2_loss),
        "stage2_data": dataclasses.asdict(config.stage2_data),
        "stage2_train": dataclasses.asdict(config.stage2_train),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
