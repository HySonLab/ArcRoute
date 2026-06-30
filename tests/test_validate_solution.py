"""Test gate for solution validation logic (from scripts/validate_solution.py).

All check functions are exercised with a tiny hand-built 3-arc / 2-vehicle
instance so no .npz or .sol file is required.

Instance layout
---------------
Nodes: 0, 1, 2  (directed cycle 0→1→2→0)
Required arcs (1-indexed, arc 0 = depot sentinel):
  arc 1: tail=0, head=1, class=1, demand=0.4, svc=2.0, trav=1.0
  arc 2: tail=1, head=2, class=1, demand=0.3, svc=1.5, trav=1.0
  arc 3: tail=2, head=0, class=2, demand=0.3, svc=1.0, trav=2.0

Arc-to-arc dead-heading adj[i,j] = shortest path from head(arc_i) to tail(arc_j):
  node distances (Floyd-Warshall on the cycle, traversal weights):
    d(0,0)=0  d(0,1)=1  d(0,2)=2
    d(1,0)=3  d(1,1)=0  d(1,2)=1
    d(2,0)=2  d(2,1)=3  d(2,2)=0
"""

import os
import sys
import io
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

EPS = 1e-6


# ── check functions (mirrors validate_solution.py) ────────────────────────────

def check_coverage(routes, n_req):
    served = sorted(int(a) for r in routes for a in r[1:-1])
    expected = list(range(1, n_req + 1))
    if served == expected:
        return True, f'all {n_req} arcs served exactly once'
    missing = sorted(set(expected) - set(served))
    extra   = sorted(set(served)   - set(expected))
    dups    = len(served) - len(set(served))
    return False, f'missing={missing} extra={extra} duplicates={dups}'


def check_capacity(routes, demands):
    ok = True
    for route in routes:
        if float(demands[route[1:-1]].sum()) > 1.0 + EPS:
            ok = False
    return ok


def check_sentinels(routes):
    return all(int(r[0]) == 0 and int(r[-1]) == 0 for r in routes)


def check_class_order(routes, clss, variant):
    if variant != 'P':
        return True
    for route in routes:
        cs = [int(clss[a]) for a in route[1:-1]]
        if any(cs[i] < cs[i - 1] for i in range(1, len(cs))):
            return False
    return True


def check_service_times(svc, req):
    return float(np.abs(svc[1:].astype(np.float64) - req[:, 4].astype(np.float64)).max()) <= EPS


def check_deadhead(adj_arc, node_dms, req):
    n     = len(req) + 1
    tails = np.array([0] + [int(req[i, 0]) for i in range(len(req))], dtype=np.int64)
    heads = np.array([0] + [int(req[i, 1]) for i in range(len(req))], dtype=np.int64)
    max_err, n_bad = 0.0, 0
    for i in range(n):
        for j in range(n):
            exp = float(node_dms[heads[i], tails[j]])
            got = float(adj_arc[i, j])
            err = abs(exp - got)
            if err > EPS:
                n_bad  += 1
                max_err = max(max_err, err)
    return n_bad == 0, max_err


def load_solution_from_string(text):
    """Parse a .sol text block (same format as save_sol()) from a string."""
    routes, T_saved, variant, instance, vehicles = [], np.zeros(3), 'P', '', None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, rest = line.partition(':')
        key, rest = key.strip(), rest.strip()
        if key == 'instance':
            instance = rest
        elif key == 'variant':
            variant = rest
        elif key == 'vehicles':
            vehicles = int(rest)
        elif key == 'T1':
            parts = rest.replace('T1:', '').replace('T2:', '').replace('T3:', '').split()
            T_saved = np.array([float(p) for p in parts[:3]])
        elif key.startswith('route'):
            routes.append(np.array([int(x) for x in rest.split()], dtype=np.int32))
    if vehicles is None:
        vehicles = len(routes)
    return routes, T_saved, variant, instance, vehicles


# ── shared synthetic fixture ──────────────────────────────────────────────────

def _make_fixture():
    demands  = np.array([0.0, 0.4, 0.3, 0.3])
    clss     = np.array([0,   1,   1,   2])
    svc      = np.array([0.0, 2.0, 1.5, 1.0])
    req      = np.array([[0, 1, 4, 1, 2.0, 1.0],
                         [1, 2, 3, 1, 1.5, 1.0],
                         [2, 0, 3, 2, 1.0, 2.0]], dtype=np.float64)
    node_dms = np.array([[0., 1., 2.],
                         [3., 0., 1.],
                         [2., 3., 0.]])
    adj_arc  = np.array([[0., 0., 1., 2.],   # arc 0 head=0
                         [3., 3., 0., 1.],   # arc 1 head=1
                         [2., 2., 3., 0.],   # arc 2 head=2
                         [0., 0., 1., 2.]])  # arc 3 head=0
    routes_ok = [np.array([0, 1, 2, 0], dtype=np.int32),
                 np.array([0, 3, 0],    dtype=np.int32)]
    return demands, clss, svc, req, node_dms, adj_arc, routes_ok


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCheckCoverage(unittest.TestCase):

    def setUp(self):
        *_, self.routes_ok = _make_fixture()

    def test_pass(self):
        ok, _ = check_coverage(self.routes_ok, n_req=3)
        self.assertTrue(ok)

    def test_missing_arc(self):
        routes = [np.array([0, 1, 2, 0], dtype=np.int32)]  # arc 3 missing
        ok, msg = check_coverage(routes, n_req=3)
        self.assertFalse(ok)
        self.assertIn('3', msg)

    def test_duplicate_arc(self):
        routes = [np.array([0, 1, 1, 2, 0], dtype=np.int32),
                  np.array([0, 3, 0],        dtype=np.int32)]
        ok, msg = check_coverage(routes, n_req=3)
        self.assertFalse(ok)
        self.assertIn('duplicates', msg)

    def test_extra_arc(self):
        routes = [np.array([0, 1, 2, 3, 4, 0], dtype=np.int32)]
        ok, msg = check_coverage(routes, n_req=3)
        self.assertFalse(ok)


class TestCheckCapacity(unittest.TestCase):

    def setUp(self):
        self.demands, *_ = _make_fixture()

    def test_within_capacity(self):
        # vehicle 1: 0.4+0.3=0.7, vehicle 2: 0.3 — both <=1.0
        routes = [np.array([0, 1, 2, 0], dtype=np.int32),
                  np.array([0, 3, 0],    dtype=np.int32)]
        self.assertTrue(check_capacity(routes, self.demands))

    def test_at_capacity(self):
        # single vehicle: 0.4+0.3+0.3=1.0 — exactly at limit
        routes = [np.array([0, 1, 2, 3, 0], dtype=np.int32)]
        self.assertTrue(check_capacity(routes, self.demands))

    def test_over_capacity(self):
        demands = np.array([0.0, 0.6, 0.6, 0.3])
        routes  = [np.array([0, 1, 2, 0], dtype=np.int32)]  # 1.2 > 1.0
        self.assertFalse(check_capacity(routes, demands))


class TestCheckSentinels(unittest.TestCase):

    def test_ok(self):
        routes = [np.array([0, 1, 2, 0]), np.array([0, 3, 0])]
        self.assertTrue(check_sentinels(routes))

    def test_bad_start(self):
        routes = [np.array([1, 2, 0])]
        self.assertFalse(check_sentinels(routes))

    def test_bad_end(self):
        routes = [np.array([0, 1, 2])]
        self.assertFalse(check_sentinels(routes))

    def test_single_arc_route(self):
        routes = [np.array([0, 1, 0])]
        self.assertTrue(check_sentinels(routes))


class TestCheckClassOrder(unittest.TestCase):

    def setUp(self):
        _, self.clss, *_ = _make_fixture()

    def test_non_decreasing_ok(self):
        routes = [np.array([0, 1, 2, 3, 0])]  # classes 1,1,2
        self.assertTrue(check_class_order(routes, self.clss, 'P'))

    def test_decreasing_fails(self):
        routes = [np.array([0, 3, 1, 0])]  # classes 2,1 — bad
        self.assertFalse(check_class_order(routes, self.clss, 'P'))

    def test_variant_u_always_passes(self):
        routes = [np.array([0, 3, 1, 0])]  # would fail for P
        self.assertTrue(check_class_order(routes, self.clss, 'U'))

    def test_single_arc_always_ok(self):
        routes = [np.array([0, 3, 0])]
        self.assertTrue(check_class_order(routes, self.clss, 'P'))


class TestCheckServiceTimes(unittest.TestCase):

    def setUp(self):
        _, _, self.svc, self.req, *_ = _make_fixture()

    def test_matching(self):
        ok = check_service_times(self.svc, self.req)
        self.assertTrue(ok)

    def test_mismatch(self):
        svc_bad = self.svc.copy()
        svc_bad[2] = 99.0
        self.assertFalse(check_service_times(svc_bad, self.req))


class TestCheckDeadhead(unittest.TestCase):

    def setUp(self):
        _, _, _, self.req, self.node_dms, self.adj_arc, _ = _make_fixture()

    def test_correct_adj(self):
        ok, max_err = check_deadhead(self.adj_arc, self.node_dms, self.req)
        self.assertTrue(ok, f'max_err={max_err}')

    def test_wrong_entry_detected(self):
        adj_bad = self.adj_arc.copy()
        adj_bad[1, 2] = 99.0  # should be 0.0
        ok, max_err = check_deadhead(adj_bad, self.node_dms, self.req)
        self.assertFalse(ok)
        self.assertGreater(max_err, 1.0)


class TestLoadSolutionFromString(unittest.TestCase):

    SOL = (
        "instance: data/foo.npz\n"
        "variant: P\n"
        "vehicles: 2\n"
        "T1: 5.000000  T2: 8.000000  T3: 12.000000\n"
        "route 1: 0 1 2 0\n"
        "route 2: 0 3 0\n"
    )

    def test_parses_variant(self):
        _, _, variant, _, _ = load_solution_from_string(self.SOL)
        self.assertEqual(variant, 'P')

    def test_parses_vehicles(self):
        _, _, _, _, vehicles = load_solution_from_string(self.SOL)
        self.assertEqual(vehicles, 2)

    def test_parses_T_values(self):
        _, T_saved, *_ = load_solution_from_string(self.SOL)
        np.testing.assert_allclose(T_saved, [5.0, 8.0, 12.0])

    def test_parses_routes(self):
        routes, *_ = load_solution_from_string(self.SOL)
        self.assertEqual(len(routes), 2)
        np.testing.assert_array_equal(routes[0], [0, 1, 2, 0])
        np.testing.assert_array_equal(routes[1], [0, 3, 0])

    def test_parses_instance_path(self):
        _, _, _, instance, _ = load_solution_from_string(self.SOL)
        self.assertEqual(instance, 'data/foo.npz')


if __name__ == '__main__':
    unittest.main()
