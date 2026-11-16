import pytest

from src.metrics import anls, corpus_anls, corpus_exact_match, exact_match, normalize


def test_exact_match_scores_one():
    assert anls("1/8/93", ["1/8/93"]) == 1.0


def test_normalization_case_and_whitespace():
    assert normalize("  Hello   \n World ") == "hello world"
    assert anls("  JOHN   smith ", ["John Smith"]) == 1.0


def test_empty_prediction_scores_zero():
    assert anls("", ["1/8/93"]) == 0.0
    assert anls(None, ["abc"]) == 0.0


def test_near_miss_gets_partial_credit():
    # one substitution in 5 chars -> NL = 0.2 -> 0.8
    assert anls("12345", ["12346"]) == pytest.approx(0.8)


def test_distance_at_or_above_threshold_scores_zero():
    assert anls("abc", ["xyz"]) == 0.0
    assert anls("ab", ["ax"]) == 0.0  # NL exactly 0.5 -> not < 0.5 -> 0


def test_multiple_golds_take_max():
    assert anls("12345", ["zzzzz", "12346", "12345"]) == 1.0
    assert anls("12345", ["zzzzz", "12346"]) == pytest.approx(0.8)


def test_single_string_gold_is_accepted():
    assert anls("abc", "abc") == 1.0


def test_empty_gold_list_raises():
    with pytest.raises(ValueError):
        anls("abc", [])


def test_corpus_scores_are_means():
    preds, golds = ["a", "12345", "zzz"], [["a"], ["12346"], ["abc"]]
    assert corpus_anls(preds, golds) == pytest.approx((1 + 0.8 + 0) / 3)
    assert corpus_exact_match(preds, golds) == pytest.approx(1 / 3)
    assert exact_match("A ", ["a"]) == 1.0


def test_corpus_length_mismatch_raises():
    with pytest.raises(ValueError):
        corpus_anls(["a"], [["a"], ["b"]])
