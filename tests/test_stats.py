"""Unit tests for eval/stats.py (plan Phase 5 statistical-rigor helpers).

Run from the project root:
    uv run python -m unittest tests.test_stats -v
"""
import unittest

import numpy as np

from eval.stats import (
    gap_to_bks,
    gap_summary,
    pairwise_wilcoxon,
    friedman,
    average_ranks,
    describe,
)


class TestGapToBKS(unittest.TestCase):
    def test_zero_when_obj_equals_bks(self):
        bks = np.array([3.0, 5.0, 8.0])
        self.assertTrue(np.allclose(gap_to_bks(bks, bks), 0.0))
        self.assertAlmostEqual(gap_summary(bks, bks)["mean_gap_pct"], 0.0)

    def test_sign_and_magnitude(self):
        bks = np.array([10.0, 20.0])
        obj = np.array([11.0, 20.0])     # 10% worse, then equal
        g = gap_to_bks(obj, bks)
        self.assertAlmostEqual(g[0], 0.1)
        self.assertAlmostEqual(g[1], 0.0)


class TestSignificanceTests(unittest.TestCase):
    def test_wilcoxon_identical_is_p1(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        _, p = pairwise_wilcoxon(a, a)
        self.assertEqual(p, 1.0)

    def test_wilcoxon_pvalue_in_unit_interval(self):
        rng = np.random.RandomState(0)
        a = rng.rand(30)
        b = a + 0.1
        _, p = pairwise_wilcoxon(a, b)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_friedman_needs_three(self):
        with self.assertRaises(ValueError):
            friedman(np.arange(5.0), np.arange(5.0))

    def test_friedman_pvalue_in_unit_interval(self):
        rng = np.random.RandomState(1)
        a, b, c = rng.rand(20), rng.rand(20), rng.rand(20)
        _, p = friedman(a, b, c)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class TestRanks(unittest.TestCase):
    def test_best_algorithm_has_lowest_rank(self):
        bks = np.array([1.0, 2.0, 3.0])
        good = bks * 1.05
        bad = bks * 1.20
        ranks = average_ranks(np.column_stack([bks, good, bad]))
        self.assertTrue(ranks[0] < ranks[1] < ranks[2])

    def test_ties_share_mean_rank(self):
        ranks = average_ranks(np.array([[5.0, 5.0, 9.0]]))
        self.assertAlmostEqual(ranks[0], 1.5)   # tie of first two -> (1+2)/2
        self.assertAlmostEqual(ranks[1], 1.5)
        self.assertAlmostEqual(ranks[2], 3.0)


class TestDescribe(unittest.TestCase):
    def test_describe_fields(self):
        d = describe([1.0, 2.0, 3.0])
        self.assertEqual(d["n"], 3)
        self.assertAlmostEqual(d["mean"], 2.0)
        self.assertEqual(d["min"], 1.0)
        self.assertEqual(d["max"], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
