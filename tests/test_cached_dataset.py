from datasets import Dataset as HFDataset

from aether_v3.data.cached_dataset import CTCCachedDataset


def test_reads_back_variable_length_rows(tmp_path):
    hf_ds = HFDataset.from_dict(
        {
            "semantic_codes": [[1, 2, 3], [4, 5]],
            "byte_target": [[65, 66], [67]],
        }
    )
    out_dir = tmp_path / "cache"
    hf_ds.save_to_disk(str(out_dir))

    ds = CTCCachedDataset(out_dir)
    assert len(ds) == 2

    codes0, target0 = ds[0]
    assert codes0.tolist() == [1, 2, 3]
    assert target0.tolist() == [65, 66]

    codes1, target1 = ds[1]
    assert codes1.tolist() == [4, 5]
    assert target1.tolist() == [67]
