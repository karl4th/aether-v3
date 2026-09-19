import torch

from aether_v3.data.collate import collate_ctc_batch


def test_pads_to_max_length_and_masks_correctly():
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([10, 20])),
        (torch.tensor([4, 5]), torch.tensor([30])),
    ]
    out = collate_ctc_batch(batch)

    assert out["semantic_codes"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert out["attention_mask"].dtype == torch.bool

    assert out["attention_mask"][0].tolist() == [True, True, True]
    assert out["semantic_codes"][0].tolist() == [1, 2, 3]

    assert out["attention_mask"][1].tolist() == [True, True, False]
    assert out["semantic_codes"][1, :2].tolist() == [4, 5]

    assert out["input_lengths"].tolist() == [3, 2]
    assert out["target_lengths"].tolist() == [2, 1]
    assert out["targets"].tolist() == [10, 20, 30]


def test_pad_value_is_a_real_vocab_id_not_a_sentinel():
    # Code id 0 is a legitimate semantic token - padding must be
    # distinguishable only via attention_mask, never by value.
    batch = [(torch.tensor([0, 0, 0]), torch.tensor([1])), (torch.tensor([0]), torch.tensor([2]))]
    out = collate_ctc_batch(batch)
    assert out["semantic_codes"][1].tolist() == [0, 0, 0]
    assert out["attention_mask"][1].tolist() == [True, False, False]


def test_single_example_batch():
    batch = [(torch.tensor([7, 8, 9]), torch.tensor([1, 2, 3]))]
    out = collate_ctc_batch(batch)
    assert out["semantic_codes"].shape == (1, 3)
    assert out["attention_mask"].all()
