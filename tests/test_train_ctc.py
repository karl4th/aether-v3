import contextlib
import json

import torch
from datasets import Dataset as HFDataset

from aether_v3.config import AetherSpeechConfig, CTCConfig
from aether_v3.data.cached_dataset import CTCCachedDataset
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.training.train_ctc import (
    JsonlLogger,
    _amp_autocast,
    build_dataloader,
    evaluate,
    targets_to_texts,
)


def test_targets_to_texts_splits_concatenated_targets():
    targets = torch.tensor([ord("h"), ord("i"), ord("!")])
    lengths = torch.tensor([2, 1])
    assert targets_to_texts(targets, lengths) == ["hi", "!"]


def test_amp_autocast_is_noop_on_cpu():
    ctx = _amp_autocast(torch.device("cpu"), torch.bfloat16)
    assert isinstance(ctx, contextlib.nullcontext)


def test_amp_autocast_is_real_autocast_for_cuda_device():
    # Only checks branch selection (device object construction doesn't
    # touch hardware) - safe on a CPU-only machine.
    ctx = _amp_autocast(torch.device("cuda", 0), torch.bfloat16)
    assert isinstance(ctx, torch.autocast)


def test_jsonl_logger_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = JsonlLogger(path)
    logger.log(step=1, loss=0.5)
    logger.log(step=2, loss=0.25)
    logger.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["step"] == 1
    assert row0["loss"] == 0.5
    assert "time" in row0


def _write_tiny_cache(tmp_path, name="cache"):
    hf_ds = HFDataset.from_dict(
        {
            "semantic_codes": [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5], [2, 3, 4]],
            "byte_target": [[0, 1], [2, 3], [4]],
        }
    )
    out_dir = tmp_path / name
    hf_ds.save_to_disk(str(out_dir))
    return out_dir


def test_build_dataloader_produces_correctly_shaped_batches(tmp_path):
    ds = CTCCachedDataset(_write_tiny_cache(tmp_path))
    loader, sampler = build_dataloader(
        ds, batch_size=3, shuffle=False, num_workers=0, distributed=False, drop_last=False
    )
    assert sampler is None
    batch = next(iter(loader))
    assert batch["semantic_codes"].shape[0] == 3
    assert batch["semantic_codes"].shape[1] == 8  # max length in this batch


def test_evaluate_runs_end_to_end_on_tiny_model(tmp_path):
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=1, num_heads=2, ffn_size=16, dropout=0.0
    )
    # upsampler_num_heads=2 to match hidden_size=8 (8 isn't divisible by
    # the default 12 heads).
    ctc_cfg = CTCConfig(
        vocab_size=6,
        blank_id=5,
        upsample_factor=2,
        upsampler_num_layers=1,
        upsampler_num_heads=2,
        upsampler_ffn_size=16,
        upsampler_dropout=0.0,
    )
    model = AetherCTCModel(speech_cfg, ctc_cfg)

    ds = CTCCachedDataset(_write_tiny_cache(tmp_path))
    loader, _ = build_dataloader(
        ds, batch_size=2, shuffle=False, num_workers=0, distributed=False, drop_last=False
    )

    metrics = evaluate(
        model, loader, torch.device("cpu"), torch.bfloat16, blank_id=ctc_cfg.blank_id
    )
    assert set(metrics) == {
        "loss",
        "ctc_loss",
        "semantic_loss",
        "wer",
        "cer",
        "examples",
    }
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["wer"], float)
    assert isinstance(metrics["cer"], float)
