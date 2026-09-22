"""Tests for the Stage 2 data pipeline v0 (stage2_cache/dataset/collate +
stage1_eval_manifest's pure helpers). No network, no real Mimi/Qwen -
`build_stage2_cache` runs a tiny real `AetherSpeechEncoder` over a
synthetic in-memory Stage 1 cache, and uses a trivial fake tokenizer
instead of downloading a real one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from datasets import Dataset

from aether_v3.config import AetherSpeechConfig, DataConfig
from aether_v3.data.stage1_eval_manifest import filter_rows_by_duration, verify_byte_alignment
from aether_v3.data.stage2_cache import (
    build_slue_sqa5_shards,
    build_stage2_cache,
    load_stage2_cache,
    save_stage2_cache,
)
from aether_v3.data.stage2_collate import collate_stage2_batch
from aether_v3.data.stage2_dataset import Stage2CachedDataset
from aether_v3.data.tokenizer import text_to_byte_ids
from aether_v3.models.aether_speech import AetherSpeechEncoder


class _FakeTokenizer:
    """Minimal stand-in satisfying `Stage2Tokenizer` - one id per character
    (offset so 0 stays free), no vocabulary file needed."""

    eos_token_id = 999

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) % 200 for c in text]}


def _fake_stage1_cache(texts: list[str], vocab_size: int) -> Dataset:
    rows = []
    for i, text in enumerate(texts):
        length = 6 + i  # varying lengths
        codes = [(i + j) % vocab_size for j in range(length)]
        rows.append({"semantic_codes": codes, "byte_target": text_to_byte_ids(text)})
    return Dataset.from_list(rows)


def test_build_stage2_cache_shapes_and_transcript_roundtrip():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=2, num_heads=2, ffn_size=16, dropout=0.0
    )
    encoder = AetherSpeechEncoder(speech_cfg)
    texts = ["HELLO WORLD", "A SHORT ONE", "THIS IS A LONGER UTTERANCE FOR TESTING"]
    stage1 = _fake_stage1_cache(texts, speech_cfg.semantic_vocab_size)

    with tempfile.TemporaryDirectory() as tmp:
        stage1_path = Path(tmp) / "stage1_test"
        stage1.save_to_disk(str(stage1_path))

        records = build_stage2_cache(
            stage1_path, encoder, _FakeTokenizer(), role="test", batch_size=2
        )

    assert len(records) == len(texts)
    for record, text in zip(records, texts, strict=True):
        assert record["transcript"] == text
        assert record["speech_states"].shape == (record["speech_length"], 8)
        assert record["target_ids"][-1].item() == _FakeTokenizer.eos_token_id


def test_stage2_cache_save_load_roundtrip():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=1, num_heads=2, ffn_size=16, dropout=0.0
    )
    encoder = AetherSpeechEncoder(speech_cfg)
    texts = ["ONE", "TWO THREE"]
    stage1 = _fake_stage1_cache(texts, speech_cfg.semantic_vocab_size)

    with tempfile.TemporaryDirectory() as tmp:
        stage1_path = Path(tmp) / "stage1_test"
        stage1.save_to_disk(str(stage1_path))
        records = build_stage2_cache(stage1_path, encoder, _FakeTokenizer(), role="test")

        cache_path = Path(tmp) / "stage2_test.pt"
        save_stage2_cache(records, cache_path)
        reloaded = load_stage2_cache(cache_path)
        assert len(reloaded) == len(records)

        dataset = Stage2CachedDataset(cache_path)
        assert len(dataset) == 2
        assert dataset[0]["transcript"] == "ONE"


def test_collate_stage2_batch_pads_and_masks_correctly():
    batch = [
        {
            "sample_id": "test_000000",
            "speech_states": torch.ones(4, 8, dtype=torch.float16),
            "speech_length": 4,
            "transcript": "A",
            "target_ids": torch.tensor([1, 2], dtype=torch.long),
        },
        {
            "sample_id": "test_000001",
            "speech_states": torch.ones(7, 8, dtype=torch.float16) * 2,
            "speech_length": 7,
            "transcript": "B",
            "target_ids": torch.tensor([3, 4, 5], dtype=torch.long),
        },
    ]
    out = collate_stage2_batch(batch)

    assert out["speech_states"].shape == (2, 7, 8)
    assert out["speech_mask"].tolist() == [
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
    ]
    assert out["target_ids"].shape == (2, 3)
    assert out["target_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert out["sample_ids"] == ["test_000000", "test_000001"]
    # padding must be zero, not garbage
    assert torch.all(out["speech_states"][0, 4:] == 0)
    assert torch.all(out["target_ids"][0, 2:] == 0)


def test_filter_rows_by_duration_keeps_only_in_bounds():
    data_cfg = DataConfig(min_audio_seconds=1.0, max_audio_seconds=5.0, sample_rate_in=16000)
    rows = [
        {
            "id": "a",
            "text": "IN BOUNDS",
            "audio": {"sampling_rate": 16000, "array": [0.0] * 16000 * 2},
        },
        {"id": "b", "text": "TOO SHORT", "audio": {"sampling_rate": 16000, "array": [0.0] * 8000}},
        {
            "id": "c",
            "text": "TOO LONG",
            "audio": {"sampling_rate": 16000, "array": [0.0] * 16000 * 10},
        },
    ]
    kept = filter_rows_by_duration(rows, data_cfg)
    assert [r["sample_id"] for r in kept] == ["a"]
    assert kept[0]["reference"] == "IN BOUNDS"
    assert kept[0]["duration"] == 2.0


def test_verify_byte_alignment_passes_on_matching_data():
    kept = [{"sample_id": "a", "reference": "HELLO", "duration": 1.0}]
    verify_byte_alignment(kept, [text_to_byte_ids("HELLO")])  # must not raise


def test_verify_byte_alignment_raises_on_mismatch():
    kept = [{"sample_id": "a", "reference": "HELLO", "duration": 1.0}]
    try:
        verify_byte_alignment(kept, [text_to_byte_ids("GOODBYE")])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on byte_target mismatch")


def test_verify_byte_alignment_raises_on_length_mismatch():
    kept = [{"sample_id": "a", "reference": "HELLO", "duration": 1.0}]
    try:
        verify_byte_alignment(kept, [])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on length mismatch")


def test_slue_builder_resamples_before_calling_current_mimi_api(tmp_path):
    class FakeMimi:
        target_sample_rate = 24000

        def encode_semantic(self, waveforms):
            assert len(waveforms[0]) == 24000
            return [torch.tensor([1, 2, 3]).numpy()]

    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    rows = [
        {
            "question_id": "q1",
            "question_audio": {"array": [0.0] * 16000, "sampling_rate": 16000},
            "raw_document_text": "The answer is Paris.",
            "answer_spans": [{"answer": "Paris"}],
        }
    ]
    count = build_slue_sqa5_shards(
        rows,
        FakeMimi(),
        AetherSpeechEncoder(speech_cfg),
        _FakeTokenizer(),
        tmp_path,
        "train",
        device="cpu",
    )
    assert count == 1
    assert len(load_stage2_cache(tmp_path / "shard-000000.pt")) == 1
