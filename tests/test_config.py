from pathlib import Path

from aether_v3.config import ExperimentConfig, load_config, save_config


def test_load_config_defaults(tmp_path: Path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.mimi.pretrained_id == "kyutai/mimi"
    assert cfg.ctc.vocab_size == 257
    assert cfg.ctc.blank_id == 256
    assert cfg.llm.model_id == "Qwen/Qwen3-4B"
    assert cfg.stage2_data.dataset_config == "sqa5"


def test_load_config_overrides_only_specified_fields(tmp_path: Path):
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(
        "aether_speech:\n  hidden_size: 128\n  num_heads: 4\ntrain:\n  batch_size: 8\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.aether_speech.hidden_size == 128
    assert cfg.aether_speech.num_heads == 4
    assert cfg.train.batch_size == 8
    # untouched fields keep their dataclass defaults
    assert cfg.aether_speech.num_layers == 8
    assert cfg.train.lr == 3e-4


def test_load_nested_stage2_config(tmp_path: Path):
    cfg_path = tmp_path / "stage2.yaml"
    cfg_path.write_text(
        "connector:\n"
        "  resampler:\n    enabled: false\n    ratio: 1\n"
        "  bridge:\n    output_dim: 1024\n"
        "llm:\n  model_id: Qwen/Qwen3-0.6B\n"
        "stage2_train:\n  max_steps: 7\n"
    )
    cfg = load_config(cfg_path)
    assert not cfg.connector.resampler.enabled
    assert cfg.connector.bridge.output_dim == 1024
    assert cfg.llm.model_id == "Qwen/Qwen3-0.6B"
    assert cfg.stage2_train.max_steps == 7


def test_save_then_load_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.batch_size = 99
    cfg.data.train_splits = ["clean/train.100"]
    out_path = tmp_path / "roundtrip.yaml"
    save_config(cfg, out_path)
    reloaded = load_config(out_path)
    assert reloaded.train.batch_size == 99
    assert reloaded.data.train_splits == ["clean/train.100"]
