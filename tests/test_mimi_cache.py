"""Unit tests for the pure/offline-computable pieces of `mimi_cache.py`.

`_extract_generator`/`extract_split`/`prepare_cache` need a live Mimi model
(network download) and a real LibriSpeech split (network download), so
they're exercised by `scripts/prepare_data.py` against
`configs/ctc_dummy.yaml` instead (an actual end-to-end smoke test), not
here.
"""

from aether_v3.config import CTCConfig, DataConfig, MimiConfig
from aether_v3.data.mimi_cache import _cache_fingerprint, _ctc_min_input_length

_UPSAMPLE_FACTOR = CTCConfig().upsample_factor


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
    fp1 = _cache_fingerprint(data_cfg, mimi_cfg, ["clean/train.100"], _UPSAMPLE_FACTOR)
    fp2 = _cache_fingerprint(data_cfg, mimi_cfg, ["clean/train.360"], _UPSAMPLE_FACTOR)
    assert fp1 != fp2


def test_cache_fingerprint_is_deterministic():
    data_cfg = DataConfig()
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    assert _cache_fingerprint(data_cfg, mimi_cfg, specs, _UPSAMPLE_FACTOR) == _cache_fingerprint(
        data_cfg, mimi_cfg, specs, _UPSAMPLE_FACTOR
    )


def test_cache_fingerprint_changes_with_duration_bounds():
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(DataConfig(max_audio_seconds=20.0), mimi_cfg, specs, _UPSAMPLE_FACTOR)
    fp2 = _cache_fingerprint(DataConfig(max_audio_seconds=30.0), mimi_cfg, specs, _UPSAMPLE_FACTOR)
    assert fp1 != fp2


def test_cache_fingerprint_changes_with_mimi_settings():
    data_cfg = DataConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(data_cfg, MimiConfig(num_quantizers=1), specs, _UPSAMPLE_FACTOR)
    fp2 = _cache_fingerprint(data_cfg, MimiConfig(num_quantizers=2), specs, _UPSAMPLE_FACTOR)
    assert fp1 != fp2


def test_cache_fingerprint_changes_with_ctc_upsample_factor():
    # A stale cache built at a different upsample factor kept a different
    # (and now-mismatched) set of CTC-feasible train examples - must not
    # be silently reused (see the comment in _extract_generator).
    data_cfg = DataConfig()
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(data_cfg, mimi_cfg, specs, 1)
    fp2 = _cache_fingerprint(data_cfg, mimi_cfg, specs, 4)
    assert fp1 != fp2


def test_cache_fingerprint_ignores_performance_only_knobs():
    # extraction_batch_size/extraction_num_workers change speed, not
    # output content - they must not invalidate an otherwise-identical
    # cache.
    mimi_cfg = MimiConfig()
    specs = ["clean/train.100"]
    fp1 = _cache_fingerprint(
        DataConfig(extraction_num_workers=1), mimi_cfg, specs, _UPSAMPLE_FACTOR
    )
    fp2 = _cache_fingerprint(
        DataConfig(extraction_num_workers=8), mimi_cfg, specs, _UPSAMPLE_FACTOR
    )
    assert fp1 == fp2
