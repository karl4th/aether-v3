"""Unit tests for the pure/offline-computable pieces of `mimi_cache.py`.

`_extract_generator`/`extract_split`/`prepare_cache` need a live Mimi model
(network download) and a real LibriSpeech split (network download), so
they're exercised by `scripts/prepare_data.py` against
`configs/ctc_dummy.yaml` instead (an actual end-to-end smoke test), not
here.
"""
from aether_v3.config import DataConfig, MimiConfig
from aether_v3.data.mimi_cache import _cache_fingerprint, _ctc_min_input_length


def test_no_repeats_needs_exactly_len_frames():
    assert _ctc_min_input_length([1, 2, 3, 4]) == 4


def test_one_adjacent_repeat_needs_one_extra_frame():
    assert _ctc_min_input_length([1, 1, 2]) == 4


def test_two_adjacent_repeats_need_two_extra_frames():
    assert _ctc_min_input_length([1, 1, 1]) == 5


def test_empty_target_needs_zero_frames():
    assert _ctc_min_input_length([]) == 0


def test_non_adjacent_repeats_need_no_extra_frames():
    assert _ctc_min_input_length([1, 2, 1, 2]) == 4


def test_cache_fingerprint_changes_with_different_splits():
    data_cfg = DataConfig()
    mimi_cfg = MimiConfig()
    fp1 = _cache_fingerprint(data_cfg, mimi_cfg, ["clean/train.100"])
    fp2 = _cache_fingerprint(data_cfg, mimi_cfg, ["clean/train.360"])
    assert fp1 != fp2


def test_cache_fingerprint_is_deterministic():
    data_cfg = DataConfig()
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    assert _cache_fingerprint(data_cfg, mimi_cfg, specs) == _cache_fingerprint(
        data_cfg, mimi_cfg, specs
    )


def test_cache_fingerprint_changes_with_duration_bounds():
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(DataConfig(max_audio_seconds=20.0), mimi_cfg, specs)
    fp2 = _cache_fingerprint(DataConfig(max_audio_seconds=30.0), mimi_cfg, specs)
    assert fp1 != fp2


def test_cache_fingerprint_changes_with_mimi_settings():
    data_cfg = DataConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(data_cfg, MimiConfig(num_quantizers=1), specs)
    fp2 = _cache_fingerprint(data_cfg, MimiConfig(num_quantizers=2), specs)
    assert fp1 != fp2


def test_cache_fingerprint_ignores_performance_only_knobs():
    # extraction_batch_size/extraction_num_workers change speed, not
    # output content - they must not invalidate an otherwise-identical
    # cache.
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(DataConfig(extraction_num_workers=1), mimi_cfg, specs)
    fp2 = _cache_fingerprint(DataConfig(extraction_num_workers=8), mimi_cfg, specs)
    assert fp1 == fp2
