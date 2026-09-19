"""`load_split`/`load_splits` need network access (HF Hub) - covered by the
`scripts/prepare_data.py` smoke test instead. Only the pure parsing logic is
unit-tested here.
"""

import pytest

from aether_v3.data.librispeech import _parse_split_spec


def test_parses_config_and_split():
    assert _parse_split_spec("clean/train.100") == ("clean", "train.100")


def test_parses_split_names_containing_dots():
    assert _parse_split_spec("other/validation.other") == ("other", "validation.other")


def test_rejects_missing_slash():
    with pytest.raises(ValueError):
        _parse_split_spec("train.100")


def test_rejects_empty_split_name():
    with pytest.raises(ValueError):
        _parse_split_spec("clean/")
