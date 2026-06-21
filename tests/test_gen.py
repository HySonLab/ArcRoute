"""Unit tests for data/gen.py:build_instance — verify benchmark instances match
the paper's "Create HDCARP instances" formulas (arXiv:2501.00852).

These tests are OSMnx-free: they feed synthetic edges + coordinates into the
pure `build_instance` function. Run from the project root:
    uv run python -m unittest tests.test_gen -v
"""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

# data/gen.py is a script under a non-package dir; load it by path.
_GEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gen.py")
_spec = importlib.util.spec_from_file_location("hdcarp_gen", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

from common.ops import import_instance


def make_strongly_connected_cycle(n, rng):
    """A directed cycle 0->1->...->n-1->0 (strongly connected) plus a few
    random chords, with random 2D coordinates."""
    coords = rng.rand(n, 2) * 100.0
    edges = [[i, (i + 1) % n] for i in range(n)]          # the cycle
    for _ in range(n):                                    # chords
        u, v = rng.randint(0, n), rng.randint(0, n)
        if u != v:
            edges.append([u, v])
    edges = np.unique(np.array(edges), axis=0)
    return edges, coords


class TestBuildInstance(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.edges, self.coords = make_strongly_connected_cycle(120, self.rng)
        self.req, self.nonreq, self.C = gen.build_instance(
            self.edges, self.coords, M=5, rng=self.rng
        )

    def _d(self, row):           # traversal column
        return row[:, 5]

    def test_columns_shape(self):
        self.assertEqual(self.req.shape[1], 6)
        self.assertEqual(self.nonreq.shape[1], 6)
        self.assertEqual(len(self.req) + len(self.nonreq), len(self.edges))

    def test_traversal_normalized(self):
        # Paper step 5: d_a = d'_a / d'_max  -> max over ALL arcs is 1.0.
        all_d = np.concatenate([self._d(self.req), self._d(self.nonreq)])
        self.assertAlmostEqual(float(all_d.max()), 1.0, places=6)
        self.assertGreaterEqual(float(all_d.min()), 0.0)

    def test_service_is_double_traversal(self):
        # Paper step 5: service = 2 * traversal (required arcs).
        self.assertTrue(np.allclose(self.req[:, 4], 2.0 * self.req[:, 5]))

    def test_demand_formula(self):
        # Paper step 6: q_a = 0.5 * d_a + 0.5 (required arcs).
        self.assertTrue(np.allclose(self.req[:, 2], 0.5 * self.req[:, 5] + 0.5))

    def test_capacity_formula(self):
        # Paper step 7: Q = sum over required arcs of (q_a / 3 + 0.5).
        ref = (self.req[:, 2] / 3.0 + 0.5).sum()
        self.assertAlmostEqual(self.C, float(ref), places=6)

    def test_priority_classes(self):
        clss = self.req[:, 3].astype(int)
        self.assertTrue(set(clss.tolist()).issubset({1, 2, 3}))
        # non-required arcs carry class 0
        self.assertTrue(np.all(self.nonreq[:, 3] == 0))

    def test_depot_on_required_arc(self):
        self.assertIn(0, self.req[:, 0].astype(int).tolist())

    def test_required_count_large(self):
        # |A| = 120 (>= 80) -> |A_r| in [60, 70]
        self.assertGreaterEqual(len(self.req), 60)
        self.assertLessEqual(len(self.req), 70)

    def test_required_count_small(self):
        rng = np.random.RandomState(1)
        edges, coords = make_strongly_connected_cycle(40, rng)
        req, nonreq, _ = gen.build_instance(edges, coords, M=2, rng=rng)
        num_arc = len(req) + len(nonreq)
        self.assertEqual(len(req), max(1, round(0.75 * num_arc)))


class TestRequiredCount(unittest.TestCase):
    def test_small_is_75_percent(self):
        rng = np.random.RandomState(0)
        self.assertEqual(gen.required_count(40, rng), 30)
        self.assertEqual(gen.required_count(20, rng), 15)

    def test_large_in_range(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            self.assertTrue(60 <= gen.required_count(150, rng) <= 70)


class TestRoundTripWithImportInstance(unittest.TestCase):
    """A generated .npz must load cleanly through common.ops.import_instance."""

    def test_npz_roundtrip(self):
        rng = np.random.RandomState(2)
        edges, coords = make_strongly_connected_cycle(120, rng)
        req, nonreq, C = gen.build_instance(edges, coords, M=5, rng=rng)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inst.npz")
            np.savez(path, req=req, nonreq=nonreq, P=3, M=5, C=C)
            dms, P, M, demands, clss, s, dd, edge_indxs = import_instance(path)
        self.assertEqual(P, [1, 2, 3])
        self.assertEqual(M, [0, 1, 2, 3, 4])
        # import_instance prepends a depot row, so length is |A_r| + 1.
        self.assertEqual(len(demands), len(req) + 1)
        self.assertEqual(dms.shape[0], len(req) + 1)
        self.assertFalse(np.isnan(dms).any())
        # demands are normalized by capacity at load time.
        self.assertTrue(np.allclose(demands[1:], req[:, 2] / C))


if __name__ == "__main__":
    unittest.main(verbosity=2)
