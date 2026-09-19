import torch

from aether_v3.data.tokenizer import BLANK_ID
from aether_v3.eval.decode import greedy_ctc_decode


def _log_probs_for_ids(ids: list[int], vocab_size: int) -> torch.Tensor:
    """Builds (1, T, V) log-probs whose per-frame argmax is exactly `ids`."""
    t = len(ids)
    logits = torch.full((1, t, vocab_size), -10.0)
    for i, tok in enumerate(ids):
        logits[0, i, tok] = 10.0
    return torch.log_softmax(logits, dim=-1)


def test_collapses_repeats_and_drops_blank():
    ids = [ord("H"), ord("H"), BLANK_ID, ord("i"), ord("i"), ord("i")]
    log_probs = _log_probs_for_ids(ids, vocab_size=257)
    [text] = greedy_ctc_decode(log_probs, input_lengths=torch.tensor([len(ids)]))
    assert text == "Hi"


def test_repeat_separated_by_blank_is_kept_twice():
    ids = [ord("l"), BLANK_ID, ord("l")]
    log_probs = _log_probs_for_ids(ids, vocab_size=257)
    [text] = greedy_ctc_decode(log_probs, input_lengths=torch.tensor([len(ids)]))
    assert text == "ll"


def test_respects_input_length_truncation():
    ids = [ord("A"), ord("B"), ord("C")]
    log_probs = _log_probs_for_ids(ids, vocab_size=257)
    [text] = greedy_ctc_decode(log_probs, input_lengths=torch.tensor([2]))
    assert text == "AB"


def test_invalid_byte_sequence_does_not_raise():
    ids = [0x80]  # lone UTF-8 continuation byte, invalid on its own
    log_probs = _log_probs_for_ids(ids, vocab_size=257)
    [text] = greedy_ctc_decode(log_probs, input_lengths=torch.tensor([1]))
    assert "�" in text


def test_custom_blank_id():
    ids = [1, 1, 0, 2]  # blank_id=0 here instead of the default 256
    log_probs = _log_probs_for_ids(ids, vocab_size=5)
    [text] = greedy_ctc_decode(log_probs, input_lengths=torch.tensor([len(ids)]), blank_id=0)
    assert text == byte_string(1) + byte_string(2)


def byte_string(byte_id: int) -> str:
    return bytes([byte_id]).decode("utf-8", errors="replace")
