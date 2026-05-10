import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiplexrag.eval_utils import (
    mrr_at_k,
    ndcg_at_k,
    parse_metric_name,
    recall_at_k,
    score_metric,
)


class RecallAtKTests(unittest.TestCase):
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 1.0)

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c", "d"}
        self.assertEqual(recall_at_k(retrieved, relevant, 4), 0.5)

    def test_zero_when_no_hits_in_topk(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 0.0)

    def test_zero_when_no_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 0.0)

    def test_topk_truncates(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"d"}
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.0)
        self.assertEqual(recall_at_k(retrieved, relevant, 4), 1.0)

    def test_duplicate_retrieved_does_not_double_count(self):
        retrieved = ["a", "a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 1.0)


class MrrAtKTests(unittest.TestCase):
    def test_first_position_is_one(self):
        self.assertEqual(mrr_at_k(["a", "b"], {"a"}, 5), 1.0)

    def test_second_position_is_half(self):
        self.assertEqual(mrr_at_k(["x", "a", "b"], {"a"}, 5), 0.5)

    def test_third_position_is_one_third(self):
        self.assertAlmostEqual(mrr_at_k(["x", "y", "a"], {"a"}, 5), 1.0 / 3.0)

    def test_no_hit_returns_zero(self):
        self.assertEqual(mrr_at_k(["x", "y", "z"], {"a"}, 3), 0.0)

    def test_only_first_hit_counts(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(mrr_at_k(retrieved, relevant, 5), 0.5)

    def test_hit_outside_topk_ignored(self):
        retrieved = ["x", "y", "z", "a"]
        relevant = {"a"}
        self.assertEqual(mrr_at_k(retrieved, relevant, 3), 0.0)


class NdcgAtKTests(unittest.TestCase):
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 3), 1.0)

    def test_single_hit_at_first_rank(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "x", "y"], {"a"}, 3), 1.0)

    def test_single_hit_at_second_rank(self):
        retrieved = ["x", "a", "y"]
        ideal = 1.0 / math.log2(2)
        actual = (1.0 / math.log2(3)) / ideal
        self.assertAlmostEqual(ndcg_at_k(retrieved, {"a"}, 3), actual)

    def test_no_hits_returns_zero(self):
        self.assertEqual(ndcg_at_k(["x", "y"], {"a"}, 2), 0.0)

    def test_empty_relevant_returns_zero(self):
        self.assertEqual(ndcg_at_k(["a"], set(), 1), 0.0)

    def test_ideal_capped_by_k(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d", "e"}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 2), 1.0)

    def test_ranking_quality_ordering(self):
        relevant = {"a"}
        first = ndcg_at_k(["a", "x", "y", "z"], relevant, 4)
        second = ndcg_at_k(["x", "a", "y", "z"], relevant, 4)
        third = ndcg_at_k(["x", "y", "a", "z"], relevant, 4)
        self.assertGreater(first, second)
        self.assertGreater(second, third)


class ParseMetricNameTests(unittest.TestCase):
    def test_parses_name_and_k(self):
        self.assertEqual(parse_metric_name("mrr@10"), ("mrr", 10))
        self.assertEqual(parse_metric_name("recall@5"), ("recall", 5))
        self.assertEqual(parse_metric_name("nDCG@20"), ("ndcg", 20))

    def test_missing_at_raises(self):
        with self.assertRaises(ValueError):
            parse_metric_name("mrr")

    def test_non_integer_k_raises(self):
        with self.assertRaises(ValueError):
            parse_metric_name("mrr@ten")


class ScoreMetricTests(unittest.TestCase):
    def test_dispatch_recall(self):
        self.assertEqual(
            score_metric("recall@2", ["a", "b", "c"], {"a", "b"}),
            recall_at_k(["a", "b", "c"], {"a", "b"}, 2),
        )

    def test_dispatch_mrr(self):
        self.assertEqual(
            score_metric("mrr@5", ["x", "a"], {"a"}),
            mrr_at_k(["x", "a"], {"a"}, 5),
        )

    def test_dispatch_ndcg(self):
        self.assertAlmostEqual(
            score_metric("ndcg@3", ["a", "x", "b"], {"a", "b"}),
            ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3),
        )

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            score_metric("precision@5", ["a"], {"a"})


if __name__ == "__main__":
    unittest.main()
