"""Unit tests for env/generator.py — verify instances match the paper's
"Create HDCARP instances" formulas (arXiv:2501.00852).

Run from the project root:
    uv run python -m unittest tests.test_generator -v
"""
import unittest
import torch

from env.generator import (
    get_sampler,
    sample_arcs,
    sample_priority_classes,
    sample_traversal_time,
    sample_service_time,
    sample_demand,
    sample_vehicle_capacity,
    generate,
)


class TestTraversalNormalization(unittest.TestCase):
    """Paper step 5: d_a = d'_a / d'_max (normalized Euclidean)."""

    def test_traversal_max_is_one(self):
        torch.manual_seed(0)
        arcs = sample_arcs(20, 20)
        trav, adj = sample_traversal_time(20, arcs, get_sampler("uniform", low=0, high=1))
        self.assertAlmostEqual(float(trav.max()), 1.0, places=5)
        self.assertGreaterEqual(float(trav.min()), 0.0)

    def test_adjacency_finite_and_nonnegative(self):
        # adj is the shortest-path time between arc endpoints (head of i ->
        # tail of j); a cross-graph path may exceed a single arc time, but it
        # must stay finite, non-negative, and NaN-free in normalized units.
        torch.manual_seed(3)
        arcs = sample_arcs(15, 15)
        trav, adj = sample_traversal_time(15, arcs, get_sampler("uniform", low=0, high=1))
        self.assertFalse(torch.isnan(adj).any())
        self.assertTrue(torch.isfinite(adj).all())
        self.assertGreaterEqual(float(adj.min()), 0.0)


class TestServiceTime(unittest.TestCase):
    """Paper step 5: service time = 2 x traversal time."""

    def test_service_is_double_traversal(self):
        trav = torch.rand(21)
        self.assertTrue(torch.allclose(sample_service_time(trav), trav * 2))


class TestDemand(unittest.TestCase):
    """Paper step 6: q_a = d_a * 0.5 + 0.5; non-required arcs have zero demand."""

    def test_demand_formula_on_required_arcs(self):
        torch.manual_seed(0)
        clss = sample_priority_classes(20)
        trav = torch.rand(clss.shape[0])
        demand = sample_demand(trav.clone(), clss)
        req = clss > 0
        self.assertTrue(torch.allclose(demand[req], trav[req] * 0.5 + 0.5))

    def test_demand_zero_exactly_at_non_required(self):
        # Regression: old code used `demand[clss] = 0` (indexing by class
        # *value*), zeroing indices {0,1,2,3} instead of the class-0 arcs.
        torch.manual_seed(0)
        clss = sample_priority_classes(20)
        trav = torch.rand(clss.shape[0])
        demand = sample_demand(trav.clone(), clss)
        zero_idx = (demand == 0).nonzero().flatten().tolist()
        nonreq_idx = (clss == 0).nonzero().flatten().tolist()
        self.assertEqual(zero_idx, nonreq_idx)


class TestVehicleCapacity(unittest.TestCase):
    """Paper step 7: Q = sum over required arcs of (q_a / 3 + 0.5)."""

    def test_capacity_sums_over_required_arcs(self):
        torch.manual_seed(0)
        clss = sample_priority_classes(20)
        demand = sample_demand(torch.rand(clss.shape[0]), clss)
        cap = sample_vehicle_capacity(demand, clss)
        ref = (demand[clss > 0] / 3 + 0.5).sum()
        self.assertTrue(torch.isclose(cap, ref))

    def test_capacity_not_constant(self):
        # Regression: the old indexing bug made Q collapse to 0.5*(n+1),
        # independent of the actual demands. Different demands must give
        # different capacities.
        torch.manual_seed(0)
        clss = sample_priority_classes(20)
        d1 = sample_demand(torch.rand(clss.shape[0]), clss)
        d2 = sample_demand(torch.rand(clss.shape[0]) * 3, clss)
        self.assertNotAlmostEqual(
            float(sample_vehicle_capacity(d1, clss)),
            float(sample_vehicle_capacity(d2, clss)),
            places=3,
        )


class TestPriorityClasses(unittest.TestCase):
    """Paper steps 3-4: classes in {1,2,3}, class 0 = non-required."""

    def test_values_in_range(self):
        torch.manual_seed(0)
        clss = sample_priority_classes(20)
        self.assertTrue(((clss >= 0) & (clss <= 3)).all())

    def test_depot_is_class_zero(self):
        torch.manual_seed(0)
        self.assertEqual(int(sample_priority_classes(20)[0]), 0)

    def test_required_ratio_small_instances(self):
        # |A| < 80  ->  |A_r| ~ 75% of |A|
        for seed in range(10):
            torch.manual_seed(seed)
            num_arc = 20
            req = int((sample_priority_classes(num_arc)[1:] > 0).sum())
            self.assertGreaterEqual(req, int(0.6 * num_arc))
            self.assertLessEqual(req, num_arc)

    def test_required_count_large_instances(self):
        # |A| > 80  ->  |A_r| in [60, 70]
        for seed in range(10):
            torch.manual_seed(seed)
            req = int((sample_priority_classes(100) > 0).sum())
            self.assertGreaterEqual(req, 55)
            self.assertLessEqual(req, 72)


class TestGenerateEndToEnd(unittest.TestCase):
    def test_generate_tensordict(self):
        torch.manual_seed(1)
        td = generate(num_loc=20, num_arc=20, num_vehicle=3)
        for key in ["clss", "demands", "capacity", "service_times",
                    "traversal_times", "adj", "num_vehicle"]:
            self.assertIn(key, td.keys())
        self.assertEqual(td.batch_size, torch.Size([1]))

    def test_no_nan(self):
        torch.manual_seed(2)
        td = generate(num_loc=20, num_arc=20, num_vehicle=3)
        self.assertFalse(torch.isnan(td["demands"]).any())
        self.assertFalse(torch.isnan(td["traversal_times"]).any())
        self.assertFalse(torch.isnan(td["adj"]).any())

    def test_relations_hold_after_generate(self):
        # service = 2 * traversal and traversal normalized to max 1, end to end.
        torch.manual_seed(5)
        td = generate(num_loc=20, num_arc=20, num_vehicle=3)
        trav = td["traversal_times"][0]
        self.assertAlmostEqual(float(trav.max()), 1.0, places=5)
        self.assertTrue(torch.allclose(td["service_times"][0], trav * 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
