"""D2 Phase 1: lexicographic best-of-K selector (T_1,T_2,T_3) — pure-function tests.

The selector picks the lexicographically-best of K sampled solutions:
`np.lexsort((obj[:,2], obj[:,1], obj[:,0]))[0]` (T_1 primary, then T_2, then T_3).
"""
import unittest

import numpy as np


def lex_best_idx(obj):
    obj = np.asarray(obj, dtype=float)
    return int(np.lexsort((obj[:, 2], obj[:, 1], obj[:, 0]))[0])


class TestLexSelector(unittest.TestCase):
    def test_lexicographic_correct(self):
        obj = np.array([[5, 9, 9], [5, 2, 9], [5, 2, 1], [7, 0, 0]], dtype=float)
        # T_1 min = 5 (rows 0,1,2); within those T_2 min = 2 (rows 1,2); T_3 min = 1 (row 2)
        self.assertEqual(lex_best_idx(obj), 2)

    def test_no_T1_regression(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            obj = rng.integers(0, 5, size=(rng.integers(2, 12), 3)).astype(float)
            sel = obj[lex_best_idx(obj)]
            self.assertEqual(sel[0], obj[:, 0].min())   # selected T_1 == global min T_1

    def test_tie_break_beats_argmin(self):
        # rows tie on T_1; old `argmin` returns the FIRST (row 0, high T_2); lex must
        # pick a row with <= T_2 (and is lexicographically <= argmin's pick).
        obj = np.array([[3, 9, 9], [3, 1, 5], [3, 1, 2], [4, 0, 0]], dtype=float)
        old = obj[obj[:, 0].argmin()]                   # row 0 -> [3,9,9]
        new = obj[lex_best_idx(obj)]                     # row 2 -> [3,1,2]
        self.assertEqual(new[0], old[0])                # same T_1
        self.assertLessEqual(new[1], old[1])            # not worse on T_2
        self.assertTrue(tuple(new) <= tuple(old))       # lexicographically <=

    def test_run_grid_uses_lex(self):
        # the eval selector source uses lexsort (guards against a regression to argmin)
        import inspect
        from eval import run_grid
        src = inspect.getsource(run_grid.RLSolver.solve)
        self.assertIn("lexsort", src)


if __name__ == "__main__":
    unittest.main()
