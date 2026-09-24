import math

from aether_v3.eval.metrics import compute_cer, compute_failure_metrics, compute_wer


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


def test_failure_metrics_detect_wh_repetition_and_empty_output():
    metrics = compute_failure_metrics(
        ["WHO ARE YOU", "HELLO THERE"],
        ["WH WH WH WH", ""],
        [2.0, 6.0],
    )
    assert metrics["short_query_examples"] == 1.0
    assert metrics["short_query_wer"] >= 1.0
    assert metrics["repetition_collapse_rate"] == 0.5
    assert metrics["empty_hypothesis_rate"] == 0.5
    assert metrics["catastrophic_failure_rate"] == 1.0


def test_failure_metrics_perfect_outputs_are_not_catastrophic():
    metrics = compute_failure_metrics(["A B", "C"], ["A B", "C"], [1.0, 5.0])
    assert metrics["short_query_wer"] == 0.0
    assert metrics["catastrophic_failure_rate"] == 0.0
    assert metrics["mean_hypothesis_reference_length_ratio"] == 1.0
    assert metrics["word_substitution_rate"] == 0.0
    assert metrics["word_deletion_rate"] == 0.0
    assert metrics["word_insertion_rate"] == 0.0
    assert metrics["under_length_hypothesis_rate"] == 0.0
    assert metrics["first_word_accuracy"] == 1.0


def test_failure_metrics_separate_word_error_types():
    metrics = compute_failure_metrics(
        ["one two three four"],
        ["one too three"],
    )

    assert metrics["word_substitution_rate"] == 0.25
    assert metrics["word_deletion_rate"] == 0.25
    assert metrics["word_insertion_rate"] == 0.0
    assert metrics["under_length_hypothesis_rate"] == 1.0
    assert metrics["first_word_accuracy"] == 1.0
