import pytest
from datasets import Dataset as HFDataset

from aether_v3.config import DataConfig
from aether_v3.data.loquacious import LoquaciousSemanticDataset, load_loquacious_split


def _dataset():
    return HFDataset.from_dict(
        {
            "sample_id": ["a", "b"],
            "semantic_codes": [[1, 2, 3], [4, 5]],
            "semantic_length": [3, 2],
            "normalized_text": ["WHO ARE YOU", "hello"],
            "audio_seconds": [2.0, 1.5],
        }
    )


def test_adapts_semantic_codes_and_derives_utf8_targets():
    dataset = LoquaciousSemanticDataset(_dataset())
    codes, target, seconds = dataset[0]
    assert dataset.lengths == (3, 2)
    assert codes.tolist() == [1, 2, 3]
    assert bytes(target.tolist()).decode() == "WHO ARE YOU"
    assert seconds == 2.0


def test_rejects_schema_and_row_length_mismatches():
    missing = HFDataset.from_dict({"semantic_codes": [[1]], "semantic_length": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        LoquaciousSemanticDataset(missing)

    invalid = HFDataset.from_dict(
        {
            "sample_id": ["bad"],
            "semantic_codes": [[1, 2]],
            "semantic_length": [3],
            "normalized_text": ["text"],
            "audio_seconds": [1.0],
        }
    )
    dataset = LoquaciousSemanticDataset(invalid)
    with pytest.raises(ValueError, match="semantic_length mismatch"):
        dataset[0]


def test_hub_loader_requires_pinned_revision():
    config = DataConfig(dataset_revision=None)
    with pytest.raises(ValueError, match="dataset_revision is required"):
        load_loquacious_split(config, "train")


def test_hub_loader_restricts_download_to_requested_split(monkeypatch, tmp_path):
    captured = {}

    def fake_load_dataset(dataset_id, **kwargs):
        captured["dataset_id"] = dataset_id
        captured.update(kwargs)
        return _dataset()

    monkeypatch.setattr("aether_v3.data.loquacious.load_dataset", fake_load_dataset)
    config = DataConfig(cache_dir=str(tmp_path))
    load_loquacious_split(config, "validation")

    assert captured["dataset_id"] == "manifestro/stage1_aether"
    assert captured["data_files"] == {"validation": "validation/*.parquet"}
    assert captured["split"] == "validation"
    assert captured["revision"] == config.dataset_revision
