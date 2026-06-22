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
    win_rate,
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


class TestWinRate(unittest.TestCase):
    """D2 Phase 6: lexicographic paired win-rate + T_1 regression count."""

    def test_win_rate_lexicographic_correct(self):
        # A vs B per instance (T_1,T_2,T_3), lower better:
        #  i0: A=[5,2,1] beats B=[5,2,9]   (lex win, T_1 tie)
        #  i1: A=[5,2,9] loses to B=[5,2,1] (lex loss, T_1 tie)
        #  i2: A=[3,0,0] == B=[3,0,0]       (tie)
        a = np.array([[5, 2, 1], [5, 2, 9], [3, 0, 0]], dtype=float)
        b = np.array([[5, 2, 9], [5, 2, 1], [3, 0, 0]], dtype=float)
        r = win_rate(a, b)
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["win_rate"], 1 / 3)
        self.assertAlmostEqual(r["loss_rate"], 1 / 3)
        self.assertAlmostEqual(r["tie_rate"], 1 / 3)
        self.assertEqual(r["t1_regression"], 0)   # T_1 never worse

    def test_t1_regression_counted(self):
        # A's T_1 is HIGHER (worse) than B's on instance 0 -> 1 regression.
        a = np.array([[6, 0, 0], [4, 0, 0]], dtype=float)
        b = np.array([[5, 9, 9], [4, 9, 9]], dtype=float)
        r = win_rate(a, b)
        self.assertEqual(r["t1_regression"], 1)

    def test_zero_t1_regression_when_t1_equal(self):
        rng = np.random.RandomState(0)
        t1 = rng.randint(1, 5, size=20)
        a = np.column_stack([t1, rng.rand(20), rng.rand(20)])
        b = np.column_stack([t1, rng.rand(20), rng.rand(20)])  # same T_1 column
        self.assertEqual(win_rate(a, b)["t1_regression"], 0)


class TestDescribe(unittest.TestCase):
    def test_describe_fields(self):
        d = describe([1.0, 2.0, 3.0])
        self.assertEqual(d["n"], 3)
        self.assertAlmostEqual(d["mean"], 2.0)
        self.assertEqual(d["min"], 1.0)
        self.assertEqual(d["max"], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
