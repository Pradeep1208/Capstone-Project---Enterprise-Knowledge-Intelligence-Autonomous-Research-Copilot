import math

from copilot.metrics import mrr_at_k, ndcg_at_k, numbers_match, precision_at_k, recall_at_k, token_f1


def test_basic_ir_metrics():
    ranked, qrels = ["a", "b", "c", "d"], {"c": 1, "z": 1}
    assert recall_at_k(ranked, qrels, 3) == 0.5
    assert precision_at_k(ranked, qrels, 4) == 0.25
    assert mrr_at_k(ranked, qrels, 4) == 1 / 3
    assert mrr_at_k(ranked, qrels, 2) == 0.0


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 2) == 1.0
    worse = ndcg_at_k(["x", "a"], {"a": 1}, 2)
    assert math.isclose(worse, 1 / math.log2(3))


def test_number_and_text_matching():
    assert numbers_match("$1577.00", "Capital expenditure was $1,577 million [1].") is True
    assert numbers_match("$1577.00", "It was $1,200 million.") is False
    assert numbers_match("Yes, it is profitable", "yes") is None
    assert token_f1("the cat sat", "the cat sat") == 1.0
