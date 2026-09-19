import math

from aether_v3.eval.metrics import compute_cer, compute_wer


def test_perfect_match_is_zero():
    assert compute_wer(["hello world"], ["hello world"]) == 0.0
    assert compute_cer(["hello world"], ["hello world"]) == 0.0


def test_one_word_substitution():
    wer = compute_wer(["the cat sat"], ["the dog sat"])
    assert math.isclose(wer, 1 / 3, rel_tol=1e-6)


def test_empty_reference_list_returns_nan():
    assert math.isnan(compute_wer([], []))
    assert math.isnan(compute_cer([], []))


def test_all_empty_strings_do_not_crash():
    # jiwer chokes on empty strings - metrics.py substitutes a lone space.
    assert compute_wer([""], [""]) == 0.0
    assert compute_cer([""], [""]) == 0.0


def test_completely_wrong_hypothesis_is_high_error():
    wer = compute_wer(["one two three"], ["completely different text"])
    assert wer >= 1.0
