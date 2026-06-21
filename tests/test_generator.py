"""Unit tests for env/generator.py — verify the on-the-fly training generator
matches the paper's "Create HDCARP instances" formulas (arXiv:2501.00852) AND
shares the sparse shortest-path metric with the benchmark loader
(common.ops.import_instance). See docs/plan/data_plan/01_phase0_restore_formula.md.

Run from the project root:
    uv run python -m unittest tests.test_generator -v
"""
import os
import tempfile
import unittest

import numpy as np
import torch
import networkx as nx

from env.generator import (
    get_sampler,
    sample_arcs,
    sample_priority_classes,
    sample_service_time,
    sample_demand,
    sample_vehicle_capacity,
    build_sparse_adj,
    generate,
)
from common.ops import import_instance


class TestPriorityClasses(unittest.TestCase):
    """Paper F2: ¼-split, balanced — each class {1,2,3} gets floor(|A|/4) arcs."""

    def test_quarter_split_count(self):
        for num_arc in (20, 40, 80, 120, 200):
            torch.manual_seed(0)
            clss = sample_priority_classes(num_arc)
            n_req = int((clss > 0).sum())
            self.assertEqual(n_req, 3 * (num_arc // 4))

    def test_classes_balanced(self):
        for seed in range(5):
            torch.manual_seed(seed)
            clss = sample_priority_classes(120)
            counts = [int((clss == c).sum()) for c in (1, 2, 3)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_ratio_scale_invariant(self):
        # Regression: HRDA's 60-70 rule made |A_r|/|A| collapse as |A| grew.
        # The ¼-split keeps it ~0.75 at every size.
        for num_arc in (20, 40, 80, 120, 200):
            torch.manual_seed(1)
            clss = sample_priority_classes(num_arc)
            ratio = int((clss > 0).sum()) / num_arc
            self.assertAlmostEqual(ratio, 0.75, delta=0.05)

    def test_values_in_range(self):
        torch.manual_seed(0)
        clss = sample_priority_classes(40)
        self.assertTrue(((clss >= 0) & (clss <= 3)).all())


class TestArcGraph(unittest.TestCase):
    """Paper F1: sparse, strongly-connected directed graph (depot = node 0)."""

    def test_strongly_connected(self):
        for seed in range(5):
            torch.manual_seed(seed)
            edges = sample_arcs(20, 40).numpy()
            g = nx.DiGraph()
            g.add_edges_from(edges.tolist())
            self.assertTrue(nx.is_strongly_connected(g))

    def test_arc_count_and_no_self_loops(self):
        torch.manual_seed(0)
        edges = sample_arcs(30, 60)
        self.assertEqual(len(edges), 60)
        self.assertFalse((edges[:, 0] == edges[:, 1]).any())
        # arcs are distinct
        pairs = {tuple(e) for e in edges.tolist()}
        self.assertEqual(len(pairs), 60)

    def test_requires_enough_arcs(self):
        with self.assertRaises(AssertionError):
            sample_arcs(num_loc=40, num_arc=20)   # |A| < n: no Hamiltonian cycle


class TestPerArcFormulas(unittest.TestCase):
    """Paper F4: service = 2*d, demand = 0.5*d + 0.5."""

    def test_service_is_double_traversal(self):
        d = torch.rand(20)
        self.assertTrue(torch.allclose(sample_service_time(d), d * 2))

    def test_demand_formula(self):
        d = torch.rand(20)
        self.assertTrue(torch.allclose(sample_demand(d), d * 0.5 + 0.5))


class TestVehicleCapacity(unittest.TestCase):
    """Paper F5: Q = (sum q_a)/3 + 0.5 — add 0.5 ONCE (not per arc)."""

    def test_capacity_formula(self):
        q = torch.rand(30) * 0.5 + 0.5
        cap = float(sample_vehicle_capacity(q))
        self.assertAlmostEqual(cap, float(q.sum()) / 3 + 0.5, places=5)

    def test_capacity_differs_from_old_buggy_formula(self):
        # Old code computed (q/3 + 0.5).sum() = sum(q)/3 + 0.5*|A_r| (~75x looser).
        q = torch.rand(30) * 0.5 + 0.5
        cap = float(sample_vehicle_capacity(q))
        old = float((q / 3 + 0.5).sum())
        self.assertGreater(abs(cap - old), 0.5 * (len(q) - 1) - 1e-6)


class TestSparseMetricMatchesTest(unittest.TestCase):
    """⭐ The KEY gate (plan §0.6): the train `adj` must be byte-identical to what
    common.ops.import_instance computes on the SAME sparse arc set — i.e. both go
    through floyd-warshall over the real arcs, NOT the old complete-Euclid metric.
    """

    def _build_arrays(self, num_loc, num_arc, seed):
        torch.manual_seed(seed)
        coords = get_sampler("uniform", low=0, high=1).sample((num_loc, 2))
        edges = sample_arcs(num_loc, num_arc)
        d_eucl = (coords[edges[:, 0]] - coords[edges[:, 1]]).pow(2).sum(-1).sqrt()
        d = d_eucl / d_eucl.max()
        req_mask = (sample_priority_classes(num_arc) > 0).numpy()
        return edges, d, req_mask

    def test_train_equals_import_instance(self):
        edges, d, req_mask = self._build_arrays(20, 40, seed=3)
        adj_train = build_sparse_adj(edges, d, req_mask).numpy()

        e = edges.numpy().astype(float)
        dd = d.numpy().astype(float)

        def cols(m):
            k = int(m.sum())
            z = np.zeros(k)
            return np.column_stack([e[m, 0], e[m, 1], z, z, z, dd[m]])

        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "i.npz")
            np.savez(p, req=cols(req_mask), nonreq=cols(~req_mask), P=3, M=5, C=1.0)
            dms = import_instance(p)[0]
        self.assertEqual(adj_train.shape, dms.shape)
        self.assertTrue(np.allclose(adj_train, dms, atol=1e-4))

    def test_adj_is_finite_and_sparse(self):
        edges, d, req_mask = self._build_arrays(20, 40, seed=7)
        adj = build_sparse_adj(edges, d, req_mask).numpy()
        self.assertTrue(np.isfinite(adj).all())     # strongly connected -> no inf
        self.assertGreaterEqual(adj.min(), 0.0)


class TestGenerateEndToEnd(unittest.TestCase):
    def test_shapes_and_depot_row(self):
        torch.manual_seed(1)
        num_arc = 40
        td = generate(num_loc=20, num_arc=num_arc, num_vehicle=3)
        n = 3 * (num_arc // 4) + 1                    # |A_r| + depot
        for key in ["clss", "demands", "service_times", "traversal_times"]:
            self.assertEqual(tuple(td[key].shape), (1, n))
        self.assertEqual(tuple(td["adj"].shape), (1, n, n))
        # depot row (index 0) is all zeros, mirroring import_instance.
        self.assertEqual(int(td["clss"][0, 0]), 0)
        self.assertEqual(float(td["demands"][0, 0]), 0.0)
        self.assertEqual(float(td["service_times"][0, 0]), 0.0)

    def test_no_nan_and_finite(self):
        torch.manual_seed(2)
        td = generate(num_loc=20, num_arc=40, num_vehicle=3)
        for key in ["demands", "traversal_times", "adj"]:
            self.assertFalse(torch.isnan(td[key]).any())
            self.assertTrue(torch.isfinite(td[key]).all())

    def test_relations_hold_after_generate(self):
        torch.manual_seed(5)
        td = generate(num_loc=20, num_arc=40, num_vehicle=3)
        trav = td["traversal_times"][0]
        # service = 2 * traversal on required arcs (depot row both zero).
        self.assertTrue(torch.allclose(td["service_times"][0], trav * 2))

    def test_demands_normalized_by_capacity(self):
        # total demand ~= 3 (since Q ~= sum(q)/3), so 3 vehicles of cap 1 ~ tight.
        torch.manual_seed(0)
        td = generate(num_loc=20, num_arc=40, num_vehicle=3)
        self.assertLess(float(td["demands"][0].sum()), 3.5)
        self.assertGreater(float(td["demands"][0].sum()), 2.0)


class TestPhase1SizeCap(unittest.TestCase):
    """Plan Phase 1: hard cap |A_r| <= 100, range support, single-size batches."""

    def test_cap_boundary(self):
        # |A|=135 -> |A_r|=3*33=99 (ok); |A|=136 -> 3*34=102 (> cap, raise).
        torch.manual_seed(0)
        td = generate(num_loc=50, num_arc=135, num_vehicle=3)
        self.assertEqual(tuple(td["adj"].shape), (1, 99 + 1, 99 + 1))
        with self.assertRaises(AssertionError):
            generate(num_loc=50, num_arc=136, num_vehicle=3)

    def test_range_support(self):
        seen = set()
        for _ in range(30):
            torch.manual_seed(len(seen) + 1)
            td = generate(num_loc=(20, 50), num_arc=(40, 120), num_vehicle=(2, 7))
            n = td["adj"].shape[1]
            self.assertLessEqual(n, 101)              # within the cap
            self.assertIn(int(td["num_vehicle"]) if not torch.is_tensor(td["num_vehicle"])
                          else int(td["num_vehicle"][0]), range(2, 8))
            seen.add(n)
        self.assertGreater(len(seen), 1)              # sizes actually vary

    def test_batch_must_be_single_size(self):
        # Same size -> collate (torch.cat) works.
        torch.manual_seed(0)
        a = generate(20, 40, 3)
        b = generate(20, 40, 3)
        torch.cat([a, b], dim=0)                       # |A_r|=30 both -> ok
        # Different size -> torch.cat fails (this is WHY we bucket by size).
        c = generate(20, 80, 3)                        # |A_r|=60
        with self.assertRaises(RuntimeError):
            torch.cat([a, c], dim=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
