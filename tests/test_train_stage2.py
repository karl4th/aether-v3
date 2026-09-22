from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from aether_v3.config import (
    AetherSpeechConfig,
    BridgeConfig,
    ConnectorConfig,
    ExperimentConfig,
    LLMConfig,
    ResamplerConfig,
)
from aether_v3.models.aether_speech_llm import AetherSpeechLLM
from aether_v3.training.train_stage2 import run_stage2_training


class TinyTokenizer:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return "answer" if ids else ""


def _model_and_config():
    speech = AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    connector = ConnectorConfig(
        resampler=ResamplerConfig(enabled=False, ratio=1, num_heads=2, ffn_size=16),
        bridge=BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=16, dropout=0.0),
    )
    llm_cfg = LLMConfig(model_id="unused", frozen=True, dtype="float32")
    llm = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
    )
    model = AetherSpeechLLM(speech, connector, llm_cfg, speech_frozen=True, llm=llm)
    cfg = ExperimentConfig(aether_speech=speech, connector=connector, llm=llm_cfg)
    cfg.stage2_train.batch_size = 1
    cfg.stage2_train.grad_accum_steps = 1
    cfg.stage2_train.max_steps = 2
    cfg.stage2_train.warmup_steps = 0
    cfg.stage2_train.log_interval = 1
    cfg.stage2_train.eval_interval = 1
    cfg.stage2_train.save_interval = 1
    cfg.stage2_train.eval_max_examples = 1
    cfg.stage2_train.generation_max_new_tokens = 2
    cfg.stage2_train.num_workers = 0
    return model, cfg


def _write_shard(path: Path):
    path.mkdir(parents=True)
    records = [
        {
            "sample_id": "q1",
            "speech_states": torch.randn(4, 8),
            "speech_length": 4,
            "prefix_ids": torch.tensor([3, 4]),
            "target_ids": torch.tensor([5, 2]),
            "references": ["answer"],
        }
    ]
    torch.save(records, path / "shard-000000.pt")


def test_complete_stage2_train_eval_checkpoint_and_resume(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    _write_shard(train)
    _write_shard(validation)
    run = tmp_path / "run"
    model, cfg = _model_and_config()
    run_stage2_training(cfg, run, train, validation, model=model, tokenizer=TinyTokenizer())
    for name in (
        "last.pt",
        "best_val_loss.pt",
        "best_answer_f1.pt",
        "best_exact_match.pt",
    ):
        assert (run / name).exists()
    assert (run / "periodic" / "step_000002.pt").exists()

    resumed_model, resumed_cfg = _model_and_config()
    resumed_cfg.stage2_train.max_steps = 3
    resumed_cfg.stage2_train.resume_from = str(run / "last.pt")
    run_stage2_training(
        resumed_cfg,
        run,
        train,
        validation,
        model=resumed_model,
        tokenizer=TinyTokenizer(),
    )
    checkpoint = torch.load(run / "last.pt", weights_only=False)
    assert checkpoint["step"] == 3
