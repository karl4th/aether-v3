from aether_v3.data.frame_batch_sampler import FrameBudgetBatchSampler


def _sampler(**overrides):
    args = {
        "lengths": [2, 3, 4, 5, 6, 7, 8, 9],
        "max_frames": 20,
        "max_examples": 4,
        "shuffle": True,
        "seed": 17,
        "bucket_size": 4,
    }
    args.update(overrides)
    return FrameBudgetBatchSampler(**args)


def test_batches_respect_padded_frame_and_example_limits():
    sampler = _sampler(shuffle=False)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(8))
    for batch in batches:
        assert len(batch) <= 4
        assert max(sampler.lengths[index] for index in batch) * len(batch) <= 20


def test_epoch_order_is_deterministic_and_changes():
    first = _sampler()
    second = _sampler()
    assert list(first) == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != list(_sampler())


def test_distributed_ranks_receive_disjoint_equal_batch_counts():
    rank0 = _sampler(rank=0, world_size=2)
    rank1 = _sampler(rank=1, world_size=2)
    batches0 = list(rank0)
    batches1 = list(rank1)
    assert len(batches0) == len(batches1)
    indices0 = {index for batch in batches0 for index in batch}
    indices1 = {index for batch in batches1 for index in batch}
    assert indices0.isdisjoint(indices1)


def test_rejects_sample_larger_than_budget():
    try:
        _sampler(lengths=[21])
    except ValueError as exc:
        assert "exceeds batch budget" in str(exc)
    else:
        raise AssertionError("expected oversized sample to be rejected")
