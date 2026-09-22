from datetime import datetime

import torch

from aether_v3.config import AetherSpeechConfig
from aether_v3.models.aether_speech import AetherSpeechEncoder
from aether_v3.training.stage2_utils import (
    answer_exact_match,
    answer_f1,
    create_run_dir,
    load_stage1_encoder,
)


def _encoder():
    cfg = AetherSpeechConfig(
        semantic_vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
    )
    return AetherSpeechEncoder(cfg)


def test_create_run_dir_uses_required_name(tmp_path):
    path = create_run_dir(tmp_path, datetime(2026, 6, 22, 18, 12, 42))
    assert path.name == "run260622-181242"
    assert (path / "periodic").is_dir()


def test_load_stage1_encoder_extracts_encoder_prefix(tmp_path):
    source = _encoder()
    path = tmp_path / "last.pt"
    torch.save({"model": {f"encoder.{k}": v for k, v in source.state_dict().items()}}, path)
    target = _encoder()
    load_stage1_encoder(path, target)
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(left, right)


def test_qa_metrics_accept_articles_and_punctuation():
    references = ["The Eiffel Tower"]
    assert answer_exact_match("eiffel tower!", references) == 1.0
    assert answer_f1("Tower", references) > 0.0
