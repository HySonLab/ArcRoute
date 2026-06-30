"""Test gate for MILP solution verification logic (from scripts/verify_milp.py).

Unit tests exercise the pure check functions using float-valued dicts so that
no real SCIP model is needed (``_get()`` short-circuits on floats).

An integration test runs the actual MILP verifier end-to-end; it is skipped
unless both PySCIPOpt and a small .npz fixture are available.
"""

import collections
import os
import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

EPS  = 1e-4
PASS = '[PASS]'
FAIL = '[FAIL]'

try:
    import pyscipopt as _pyscipopt
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False

# Smallest available fixture (20–30 arcs, loads quickly).
_FIXTURE = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'osm_train', 'new_york', '20_30.npz'
)


# ── helpers copied from verify_milp.py ───────────────────────────────────────

def _get(model, var):
    if isinstance(var, float):
        return var
    return model.getVal(var)


def _eulerian_check(model, m, lv, x_by_level, y_vars, req_by_level, A_prime, dummy):
    MG = nx.MultiDiGraph()

    for a in req_by_level[lv]:
        if _get(model, x_by_level.get((m, lv, a), 0.0)) > 0.5:
            MG.add_edge(a[0], a[1], arc=a, kind='service')

    for a in A_prime:
        cnt = int(round(_get(model, y_vars.get((m, a, lv), 0.0))))
        for _ in range(cnt):
            MG.add_edge(a[0], a[1], arc=a, kind='deadhead')

    if MG.number_of_edges() == 0:
        return True, None, None

    bad = [(v, MG.in_degree(v), MG.out_degree(v))
           for v in MG.nodes if MG.in_degree(v) != MG.out_degree(v)]
    if bad:
        return False, f'unbalanced nodes (in!=out): {bad[:5]}', None

    if nx.is_weakly_connected(MG):
        return True, None, None

    comps = list(nx.weakly_connected_components(MG))
    phantom = []
    for comp in comps:
        if dummy in comp:
            continue
        has_service = any(d.get('kind') == 'service'
                          for _u, _v, d in MG.edges(comp, data=True))
        if has_service:
            return (False,
                    f'disconnected component {sorted(comp)} contains a service arc '
                    f'(genuine sub-tour)', None)
        phantom.append(sorted(comp))

    warn = f'phantom deadhead loop(s) disconnected from dummy: {phantom}'
    return True, None, warn


def check_coverage(model, x_by_level, req_arcs, M, levels):
    lines, ok = [], True
    for a in req_arcs:
        total = sum(_get(model, x_by_level.get((m, lv, a), 0.0))
                    for m in M for lv in levels)
        if abs(total - 1.0) > EPS:
            ok = False
            lines.append(f'  {FAIL} arc {a}: served {total:.3f} times (expected 1)')
    if ok:
        lines.append(f'  {PASS} all {len(req_arcs)} required arcs served exactly once')
    return ok, lines


def check_capacity(model, x_by_level, req_arcs, M, levels, q, Q):
    lines, ok = [], True
    for m in M:
        load = sum(q[a] * _get(model, x_by_level.get((m, lv, a), 0.0))
                   for a in req_arcs for lv in levels)
        over = load > Q + EPS
        ok  &= not over
        tag  = FAIL if over else PASS
        lines.append(f'  {tag} vehicle {m}: load={load:.4f} / Q={Q:.4f}')
    return ok, lines


# ── tests: check_coverage ────────────────────────────────────────────────────

class TestMILPCheckCoverage(unittest.TestCase):
    """Directed 3-arc cycle; 2 vehicles; variant-P style x_by_level."""

    REQ  = [(0, 1), (1, 2), (2, 0)]
    M    = [0, 1]
    LEVS = [1, 2]

    def _x_ok(self):
        # vehicle 0 covers (0,1) and (1,2) at level 1; vehicle 1 covers (2,0) at level 2
        return {
            (0, 1, (0, 1)): 1.0,
            (0, 1, (1, 2)): 1.0,
            (1, 2, (2, 0)): 1.0,
        }

    def test_all_served_passes(self):
        ok, _ = check_coverage(None, self._x_ok(), self.REQ, self.M, self.LEVS)
        self.assertTrue(ok)

    def test_missing_arc_fails(self):
        x = dict(self._x_ok())
        del x[(0, 1, (0, 1))]
        ok, lines = check_coverage(None, x, self.REQ, self.M, self.LEVS)
        self.assertFalse(ok)
        self.assertTrue(any('(0, 1)' in l for l in lines))

    def test_arc_served_twice_fails(self):
        x = dict(self._x_ok())
        x[(1, 1, (0, 1))] = 1.0  # vehicle 1 also serves (0,1) — duplicate
        ok, lines = check_coverage(None, x, self.REQ, self.M, self.LEVS)
        self.assertFalse(ok)
        self.assertTrue(any('2.000' in l or '2.0' in l for l in lines))

    def test_empty_dict_all_fail(self):
        ok, lines = check_coverage(None, {}, self.REQ, self.M, self.LEVS)
        self.assertFalse(ok)
        self.assertEqual(len([l for l in lines if FAIL in l]), len(self.REQ))


# ── tests: check_capacity ────────────────────────────────────────────────────

class TestMILPCheckCapacity(unittest.TestCase):

    REQ  = [(0, 1), (1, 2), (2, 0)]
    M    = [0]
    LEVS = [1, 2]
    Q    = 2.0
    q    = {(0, 1): 1.0, (1, 2): 0.9, (2, 0): 1.0}

    def test_within_capacity(self):
        x = {(0, 1, (0, 1)): 1.0, (0, 1, (1, 2)): 1.0}  # load = 1.9
        ok, _ = check_capacity(None, x, self.REQ, self.M, self.LEVS, self.q, self.Q)
        self.assertTrue(ok)

    def test_at_capacity(self):
        x = {(0, 1, (0, 1)): 1.0, (0, 2, (2, 0)): 1.0}  # load = 2.0 exactly
        ok, _ = check_capacity(None, x, self.REQ, self.M, self.LEVS, self.q, self.Q)
        self.assertTrue(ok)

    def test_over_capacity(self):
        # all three arcs on one vehicle → load = 2.9 > Q=2.0
        x = {(0, 1, (0, 1)): 1.0, (0, 1, (1, 2)): 1.0, (0, 2, (2, 0)): 1.0}
        ok, lines = check_capacity(None, x, self.REQ, self.M, self.LEVS, self.q, self.Q)
        self.assertFalse(ok)
        self.assertTrue(any(FAIL in l for l in lines))

    def test_empty_vehicle_has_zero_load(self):
        ok, lines = check_capacity(None, {}, self.REQ, self.M, self.LEVS, self.q, self.Q)
        self.assertTrue(ok)
        self.assertIn('0.0000', lines[0])


# ── tests: _eulerian_check ───────────────────────────────────────────────────

class TestEulerianCheck(unittest.TestCase):
    """Directed 3-arc cycle 0→1→2→0 (single level, no deadhead arcs)."""

    REQ_BY_LV = {1: [(0, 1), (1, 2), (2, 0)]}
    A_PRIME   = []
    DUMMY     = 'v0'

    def test_balanced_cycle_is_eulerian(self):
        # All 3 arcs served: balanced directed cycle → traversable
        x = {(0, 1, (0, 1)): 1.0, (0, 1, (1, 2)): 1.0, (0, 1, (2, 0)): 1.0}
        ok, err, warn = _eulerian_check(None, 0, 1, x, {}, self.REQ_BY_LV, self.A_PRIME, self.DUMMY)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_incomplete_path_is_unbalanced(self):
        # Only (0,1) and (1,2): nodes 0 (out>in) and 2 (in>out) are unbalanced
        x = {(0, 1, (0, 1)): 1.0, (0, 1, (1, 2)): 1.0}
        ok, err, warn = _eulerian_check(None, 0, 1, x, {}, self.REQ_BY_LV, self.A_PRIME, self.DUMMY)
        self.assertFalse(ok)
        self.assertIn('unbalanced', err)

    def test_empty_vehicle_is_ok(self):
        ok, err, warn = _eulerian_check(None, 0, 1, {}, {}, self.REQ_BY_LV, self.A_PRIME, self.DUMMY)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIsNone(warn)

    def test_phantom_deadhead_loop_warns_not_fails(self):
        # Service cycle 0→1→2→0 connected to dummy via dummy↔0 deadhead.
        # Deadhead loop 3→4→3 is disconnected from dummy and has no service arcs
        # → phantom warning, not failure.
        A_prime_loop = [(self.DUMMY, 0), (0, self.DUMMY), (3, 4), (4, 3)]
        x = {(0, 1, (0, 1)): 1.0, (0, 1, (1, 2)): 1.0, (0, 1, (2, 0)): 1.0}
        y = {
            (0, (self.DUMMY, 0), 1): 1.0, (0, (0, self.DUMMY), 1): 1.0,
            (0, (3, 4), 1): 1.0,          (0, (4, 3), 1): 1.0,
        }
        ok, err, warn = _eulerian_check(None, 0, 1, x, y, self.REQ_BY_LV, A_prime_loop, self.DUMMY)
        self.assertTrue(ok, f'expected ok but err={err}')
        self.assertIsNone(err)
        self.assertIsNotNone(warn)
        self.assertIn('phantom', warn)

    def test_disconnected_with_service_arc_fails(self):
        # Main component: dummy↔0 (deadhead only, no service arcs).
        # Sub-tour component: balanced service cycle 1→2→3→1 disconnected from dummy.
        # The sub-tour component has service arcs → genuine sub-tour → FAIL.
        A_prime_dh = [(self.DUMMY, 0), (0, self.DUMMY)]
        req_by_lv_sub = {1: [(1, 2), (2, 3), (3, 1)]}
        x = {(0, 1, (1, 2)): 1.0, (0, 1, (2, 3)): 1.0, (0, 1, (3, 1)): 1.0}
        y = {(0, (self.DUMMY, 0), 1): 1.0, (0, (0, self.DUMMY), 1): 1.0}
        ok, err, warn = _eulerian_check(None, 0, 1, x, y, req_by_lv_sub, A_prime_dh, self.DUMMY)
        self.assertFalse(ok, 'expected sub-tour detection to fail')
        self.assertIn('sub-tour', err)


# ── integration test (requires PySCIPOpt + fixture) ──────────────────────────

@unittest.skipUnless(HAS_SCIP, 'pyscipopt not installed')
class TestVerifyMILPIntegration(unittest.TestCase):
    """Solve the MILP on a tiny hand-built instance and verify coverage + capacity."""

    # Same directed 4-arc cycle used in test_lp.py: 0→1→2→3→0, 2 classes.
    @classmethod
    def setUpClass(cls):
        from solvers.lp import solve_milp_p, build_graph
        es = {
            'req': np.array([
                [0, 1, 1, 1, 2, 1],
                [2, 3, 1, 1, 2, 1],
                [1, 2, 1, 2, 2, 1],
                [3, 0, 1, 2, 2, 1],
            ], dtype=np.float64),
            'nonreq': np.zeros((0, 6), dtype=np.float64),
            'P': np.int64(2),
            'M': np.int64(2),
            'C': np.float64(4.0),
        }
        cls.G = build_graph(es)
        cls.Tvec, cls.model, cls.vd = solve_milp_p(
            es, time_limit=60, threads=2, verbose=False, return_model=True
        )

    def test_feasible_solution_found(self):
        self.assertIsNotNone(self.Tvec, 'MILP found no feasible solution within time limit')

    def test_variant_p_coverage(self):
        if self.Tvec is None:
            self.skipTest('no feasible solution')
        G = self.G
        M       = list(range(G['M']))
        classes = list(range(1, G['P'] + 1))
        x_by_level = {(m, k, a): self.vd['x'][(m, a)]
                      for m in M for k in classes for a in G['req_by_class'][k]}
        ok, lines = check_coverage(self.model, x_by_level, G['req_arcs'], M, classes)
        self.assertTrue(ok, '\n'.join(lines))

    def test_variant_p_capacity(self):
        if self.Tvec is None:
            self.skipTest('no feasible solution')
        G = self.G
        M       = list(range(G['M']))
        classes = list(range(1, G['P'] + 1))
        x_by_level = {(m, k, a): self.vd['x'][(m, a)]
                      for m in M for k in classes for a in G['req_by_class'][k]}
        ok, lines = check_capacity(self.model, x_by_level, G['req_arcs'], M, classes,
                                   G['q'], G['Q'])
        self.assertTrue(ok, '\n'.join(lines))


if __name__ == '__main__':
    unittest.main()
