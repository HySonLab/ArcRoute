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
import networkx as nx

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
        # Paper F5: Q = (sum over required arcs of q_a) / 3 + 0.5  (add 0.5 ONCE).
        ref = self.req[:, 2].sum() / 3.0 + 0.5
        self.assertAlmostEqual(self.C, float(ref), places=6)

    def test_capacity_differs_from_old_buggy_formula(self):
        # Old code added 0.5 per required arc -> ~75x too loose.
        old = float((self.req[:, 2] / 3.0 + 0.5).sum())
        self.assertGreater(abs(self.C - old), 0.5 * (len(self.req) - 1) - 1e-6)

    def test_priority_classes(self):
        clss = self.req[:, 3].astype(int)
        self.assertTrue(set(clss.tolist()).issubset({1, 2, 3}))
        # non-required arcs carry class 0
        self.assertTrue(np.all(self.nonreq[:, 3] == 0))

    def test_classes_balanced(self):
        clss = self.req[:, 3].astype(int)
        counts = [int((clss == c).sum()) for c in (1, 2, 3)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_depot_on_required_arc(self):
        self.assertIn(0, self.req[:, 0].astype(int).tolist())

    def test_required_count_quarter_split(self):
        # Paper F2: |A_r| = 3*floor(|A|/4).
        num_arc = len(self.req) + len(self.nonreq)
        self.assertEqual(len(self.req), 3 * (num_arc // 4))

    def test_required_count_small(self):
        rng = np.random.RandomState(1)
        edges, coords = make_strongly_connected_cycle(40, rng)
        req, nonreq, _ = gen.build_instance(edges, coords, M=2, rng=rng)
        num_arc = len(req) + len(nonreq)
        self.assertEqual(len(req), 3 * (num_arc // 4))


class TestRequiredCount(unittest.TestCase):
    def test_quarter_split(self):
        rng = np.random.RandomState(0)
        self.assertEqual(gen.required_count(40, rng), 30)    # 3*10
        self.assertEqual(gen.required_count(20, rng), 15)    # 3*5
        self.assertEqual(gen.required_count(120, rng), 90)   # 3*30

    def test_scale_invariant_ratio(self):
        # No more 60-70 collapse: ratio stays ~0.75 at every size.
        for num_arc in (40, 80, 120, 200):
            ratio = gen.required_count(num_arc) / num_arc
            self.assertAlmostEqual(ratio, 0.75, delta=0.05)

    def test_bucket_cap(self):
        # Phase 1: buckets with |A_r|=3*floor(|A|/4) <= 100 are kept; larger ones
        # (|A| >= 140 -> |A_r| >= 105) are excluded.
        keep = [b for b in range(30, 201, 10) if gen.required_count(b) <= 100]
        self.assertTrue(all(gen.required_count(b) <= 100 for b in keep))
        self.assertNotIn(140, keep)                   # 3*35 = 105 > 100
        self.assertIn(130, keep)                      # 3*32 = 96 <= 100


class TestDensityMetadata(unittest.TestCase):
    """Phase 2: .npz records density d = |A|/|V|, and import_instance still loads
    cleanly despite the extra key."""

    def test_d_in_npz_and_loader_ignores_it(self):
        rng = np.random.RandomState(2)
        edges, coords = make_strongly_connected_cycle(120, rng)
        req, nonreq, C = gen.build_instance(edges, coords, M=5, rng=rng)
        num_arc = len(req) + len(nonreq)
        d = num_arc / len(coords)
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "inst.npz")
            np.savez(path, req=req, nonreq=nonreq, P=3, M=5, C=C, d=d)
            loaded = np.load(path)
            self.assertIn("d", loaded.files)
            self.assertAlmostEqual(float(loaded["d"]), d, places=6)
            # extra metadata must not break the loader.
            dms = import_instance(path)[0]
            self.assertFalse(np.isnan(dms).any())


class TestFleetSweep(unittest.TestCase):
    """Phase 3: fleet M sweep. Capacity Q must be INDEPENDENT of M (paper F5)."""

    def test_capacity_invariant_to_M(self):
        # Same graph + same rng stream, different M -> identical C.
        edges, coords = make_strongly_connected_cycle(120, np.random.RandomState(0))
        _, _, c1 = gen.build_instance(edges, coords, M=1, rng=np.random.RandomState(7))
        _, _, c10 = gen.build_instance(edges, coords, M=10, rng=np.random.RandomState(7))
        self.assertAlmostEqual(c1, c10, places=9)

    def test_vehicles_cli_accepts_multiple(self):
        # --vehicles is nargs='+' with a default sweep; no more choices=[2,5].
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--vehicles", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10])
        self.assertEqual(p.parse_args(["--vehicles", "1", "3", "7"]).vehicles, [1, 3, 7])
        self.assertEqual(p.parse_args([]).vehicles, [1, 2, 3, 5, 7, 10])


class TestSyntheticTopology(unittest.TestCase):
    """Phase 4: OOD topologies (unit_square, cluster) share F2-F5 with OSM and
    load cleanly. All OSMnx-free."""

    def test_strongly_connected_and_quarter_split(self):
        for topo in ("unit_square", "cluster"):
            rng = np.random.RandomState(0)
            edges, coords = gen._TOPOLOGIES[topo](40, 2.0, rng)
            g = nx.DiGraph()
            g.add_edges_from(edges.tolist())
            self.assertTrue(nx.is_strongly_connected(g), topo)
            req, nonreq, C = gen.build_instance(edges, coords, M=5, rng=rng)
            num_arc = len(req) + len(nonreq)
            self.assertEqual(len(req), 3 * (num_arc // 4))           # ¼-split
            counts = [int((req[:, 3] == c).sum()) for c in (1, 2, 3)]
            self.assertLessEqual(max(counts) - min(counts), 1)        # balanced
            self.assertFalse(np.isnan(req).any())

    def test_physics_identical_across_topologies(self):
        sources = {
            "cycle": make_strongly_connected_cycle(50, np.random.RandomState(1)),
            "unit_square": gen.make_unit_square(50, 2.0, np.random.RandomState(1)),
            "cluster": gen.make_cluster(50, 2.0, np.random.RandomState(1)),
        }
        for name, (edges, coords) in sources.items():
            req, _, C = gen.build_instance(edges, coords, M=5, rng=np.random.RandomState(2))
            self.assertTrue(np.allclose(req[:, 4], 2 * req[:, 5]), name)        # service=2d
            self.assertTrue(np.allclose(req[:, 2], 0.5 * req[:, 5] + 0.5), name)  # demand
            self.assertAlmostEqual(C, req[:, 2].sum() / 3 + 0.5, places=6)        # F5

    def test_gen_synth_roundtrip_and_topology_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            path, n_arc = gen.gen_synth("cluster", 40, 2.0, 5, t,
                                        rng=np.random.RandomState(3))
            loaded = np.load(path)
            self.assertEqual(str(loaded["topology"]), "cluster")
            self.assertIn("d", loaded.files)
            dms = import_instance(path)[0]                            # loads cleanly
            self.assertFalse(np.isnan(dms).any())

    @unittest.skipUnless(importlib.util.find_spec("osmnx"), "osmnx not installed")
    def test_osm_branch_importable(self):
        self.assertTrue(callable(gen.load_osm_graph))


class TestTightnessMetadata(unittest.TestCase):
    """Phase 5: .npz carries full metadata (d, M, topology, tau, n_req); tau =
    sum(q)/(M*Q) decreases as M grows (more vehicles -> looser capacity)."""

    def _save(self, M, t):
        edges, coords = gen.make_unit_square(50, 2.0, np.random.RandomState(1))
        req, nonreq, C = gen.build_instance(edges, coords, M=M, rng=np.random.RandomState(2))
        path = gen._save_instance(os.path.join(t, f"m{M}"), req, nonreq, M, C,
                                  "unit_square", rng=np.random.RandomState(0))
        return path, req, C

    def test_metadata_complete_and_loader_ok(self):
        with tempfile.TemporaryDirectory() as t:
            path, req, C = self._save(5, t)
            meta = np.load(path)
            for key in ("d", "M", "topology", "tau", "n_req"):
                self.assertIn(key, meta.files)
            self.assertEqual(int(meta["n_req"]), len(req))
            self.assertAlmostEqual(float(meta["tau"]), req[:, 2].sum() / (5 * C), places=6)
            # extra metadata must not break the loader.
            self.assertFalse(np.isnan(import_instance(path)[0]).any())

    def test_tau_decreases_with_fleet(self):
        with tempfile.TemporaryDirectory() as t:
            tau1 = float(np.load(self._save(1, t)[0])["tau"])
            tau10 = float(np.load(self._save(10, t)[0])["tau"])
            self.assertGreater(tau1, tau10)


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
