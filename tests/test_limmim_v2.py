import json
from pathlib import Path

import numpy as np

from aether_v3.config import MimiConfig
from aether_v3.data.limmim_v2 import build_limmim_v2_split


class FakeMimi:
    target_sample_rate = 16000

    def encode_semantic(self, waveforms):
        return [np.arange(max(1, len(waveform) // 1600), dtype=np.int64) for waveform in waveforms]


def _rows():
    return [
        {
            "id": f"id-{index}",
            "speaker_id": 1,
            "chapter_id": 2,
            "file": f"{index}.flac",
            "source_split": "clean/train.100",
            "text": text,
            "audio": {"array": np.zeros(samples, dtype=np.float32), "sampling_rate": 16000},
        }
        for index, (text, samples) in enumerate((("HELLO", 3200), ("WORLD", 4800), ("AGAIN", 1600)))
    ]


def test_build_limmim_v2_preserves_all_rows_and_exact_transcripts(tmp_path: Path):
    output = tmp_path / "train"
    manifest = build_limmim_v2_split(
        _rows(),
        MimiConfig(),
        output,
        "train",
        ["clean/train.100"],
        device="cpu",
        batch_size=2,
        num_workers=0,
        shard_size=2,
        mimi=FakeMimi(),
    )

    import pyarrow.parquet as pq

    rows = []
    for path in sorted(output.glob("part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    assert manifest["examples"] == len(rows) == 3
    assert rows[0]["transcript"] == "HELLO"
    assert rows[0]["source_split"] == "clean/train.100"
    assert bytes(rows[0]["byte_target"]).decode("utf-8") == "HELLO"
    assert rows[0]["semantic_length"] == len(rows[0]["semantic_codes"]) == 2
    assert json.loads((output / "_COMPLETE.json").read_text())["filters"] == []

    second = build_limmim_v2_split(
        _rows(),
        MimiConfig(),
        output,
        "train",
        ["clean/train.100"],
        device="cpu",
        num_workers=0,
        mimi=FakeMimi(),
    )
    assert second == manifest
