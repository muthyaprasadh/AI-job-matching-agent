import math

from scripts.evaluate_retrieval import _metrics


def test_retrieval_metrics_use_relevant_count_and_discounted_gain():
    recall, mrr, ndcg = _metrics(
        [("c", 2.0), ("b", 1.0), ("a", 0.0)],
        {"a": 1, "b": 0, "c": 1},
        k=2,
    )

    assert recall == 0.5
    assert mrr == 1.0
    assert math.isclose(ndcg, 1 / (1 + 1 / math.log2(3)))
