import os
import sys
import unittest
from collections import Counter
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.ils_operators import perturbate, accept, routes_to_action
from utils.local_search import ls
from utils.ops import gen_tours, deserialize_tours_batch
from solvers.evaluator import evaluate_batch
from solvers.meta import BaseHCARP, ILSHCARP, lex_sort, lex_argbest, lex_key


def make_symmetric_adj(n, seed=0):
    """Random symmetric arc-to-arc matrix with zero diagonal (n includes depot)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float64)


def arc_multiset(routes):
    """Counter of all non-depot arcs across a route list."""
    return Counter(int(a) for r in routes for a in np.asarray(r) if int(a) != 0)


# --------------------------------------------------------------------------- #
# instance factories
# --------------------------------------------------------------------------- #
def make_vars(n=7, seed=5):
    """6 arcs + depot; classes 1,1,2,2,3,3; equal demands."""
    adj = make_symmetric_adj(n, seed=seed)
    clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)[:n]
    demands = np.full(n, 0.3, dtype=np.float64)
    demands[0] = 0.0
    return adj, clss, demands


def make_instance(nv=3, seed=5):
    """Build a BaseHCARP-derived solver with attributes set directly (no .npz)."""
    n = 7  # 6 arcs + depot
    adj, clss, demands = make_vars(n=n, seed=seed)
    s = np.zeros(n, dtype=np.float64)         # service times
    d = np.zeros(n, dtype=np.float64)         # traversal
    al = ILSHCARP(strength=2)
    al.dms = adj
    al.P = [1, 2, 3]
    al.M = list(range(nv))
    al.demands = demands.astype(np.float32)
    al.clss = clss.astype(np.int32)
    al.s = s
    al.d = d
    al.nv = nv
    al.nseq = n
    al.max_len = al.nseq + (al.nv - 2)
    al.vars = {
        'adj': al.dms,
        'service_time': al.s,
        'clss': al.clss,
        'demand': al.demands,
        'nv': al.nv,
    }
    al.has_instance = True
    return al


# --------------------------------------------------------------------------- #
# 1a. perturbate
# --------------------------------------------------------------------------- #
class TestPerturbate(unittest.TestCase):
    def setUp(self):
        self.adj, self.clss, self.demands = make_vars()
        self.capacity = 1.0
        # two routes, each load 0.9 <= 1.0
        self.routes = [
            np.array([0, 1, 3, 5, 0], dtype=np.int32),
            np.array([0, 2, 4, 6, 0], dtype=np.int32),
        ]

    def test_perturbate_conserves_arcs(self):
        rng = np.random.default_rng(0)
        before = arc_multiset(self.routes)
        for _ in range(20):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            self.assertEqual(arc_multiset(out), before)

    def test_perturbate_capacity_feasible(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            for r in out:
                self.assertLessEqual(self.demands[np.asarray(r)].sum(),
                                     self.capacity + 1e-9)

    def test_perturbate_priority_order_P(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            for r in out:
                interior = [int(a) for a in np.asarray(r)[1:-1]]
                cls_seq = [int(self.clss[a]) for a in interior]
                self.assertEqual(cls_seq, sorted(cls_seq),
                                 msg=f"route {interior} not class-sorted")

    def test_perturbate_U_no_sort(self):
        rng = np.random.default_rng(3)
        before = arc_multiset(self.routes)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'U', rng, strength=3)
            self.assertEqual(arc_multiset(out), before)
            for r in out:
                self.assertLessEqual(self.demands[np.asarray(r)].sum(),
                                     self.capacity + 1e-9)

    def test_perturbate_does_not_mutate_input(self):
        rng = np.random.default_rng(4)
        snapshot = [r.copy() for r in self.routes]
        perturbate(self.routes, self.adj, self.demands, self.clss,
                   self.capacity, 'P', rng, strength=3)
        for orig, snap in zip(self.routes, snapshot):
            self.assertTrue(np.array_equal(orig, snap))

    def test_perturbate_strength_zero_returns_copy(self):
        rng = np.random.default_rng(5)
        out = perturbate(self.routes, self.adj, self.demands, self.clss,
                         self.capacity, 'P', rng, strength=0)
        self.assertEqual(arc_multiset(out), arc_multiset(self.routes))
        for r, o in zip(self.routes, out):
            self.assertTrue(np.array_equal(np.asarray(r), np.asarray(o)))

    def test_perturbate_single_route(self):
        rng = np.random.default_rng(6)
        routes = [np.array([0, 1, 2, 3, 0], dtype=np.int32)]
        before = arc_multiset(routes)
        for _ in range(20):
            out = perturbate(routes, self.adj, self.demands, self.clss,
                             self.capacity, 'U', rng, strength=2)
            # arcs conserved and still in the single route
            self.assertEqual(arc_multiset(out), before)
            self.assertEqual(len(out), 1)


# --------------------------------------------------------------------------- #
# 1b. accept
# --------------------------------------------------------------------------- #
class TestAccept(unittest.TestCase):
    def test_accept_best_mode(self):
        self.assertTrue(accept(0.0, 1.0, 0, 100, mode='best'))   # better
        self.assertTrue(accept(1.0, 1.0, 0, 100, mode='best'))   # equal
        self.assertFalse(accept(1.0, 0.0, 0, 100, mode='best'))  # worse

    def test_accept_sa_always_accepts_better(self):
        for it in range(0, 100, 10):
            self.assertTrue(accept(0.0, 5.0, it, 100, mode='sa'))
        # equal also accepted
        self.assertTrue(accept(2.0, 2.0, 50, 100, mode='sa'))

    def test_accept_sa_sometimes_rejects_worse(self):
        np.random.seed(0)
        n_accept = sum(
            accept(0.0, -10.0, 100, 100, mode='sa',
                   T_init=1e-3, T_final=1e-6)
            for _ in range(100)
        )
        # T -> T_final = 1e-6, delta = -10 => prob ~ 0 => essentially never accept
        self.assertLessEqual(n_accept, 1)


# --------------------------------------------------------------------------- #
# 1c. routes_to_action
# --------------------------------------------------------------------------- #
class TestRoutesToAction(unittest.TestCase):
    def test_roundtrip(self):
        routes = [
            np.array([0, 1, 2, 0], dtype=np.int32),
            np.array([0, 3, 4, 0], dtype=np.int32),
        ]
        action = routes_to_action(routes, nseq=10)
        self.assertTrue(np.array_equal(gen_tours(action), routes)
                        or arc_multiset(gen_tours(action)) == arc_multiset(routes))
        # explicit separator structure: [1,2,0,3,4, ...padding]
        self.assertEqual(action[0], 1)
        self.assertEqual(action[2], 0)
        self.assertEqual(action[3], 3)


# --------------------------------------------------------------------------- #
# 1d. lex_sort / lex_argbest / lex_key — exact lexicographic ordering.
#     Regression guard: ranking must never go back through a folded weighted
#     scalar (Horner encoding obj = sum_p T_p * C**(P-1-p)); the fixtures
#     below include REAL T-vector pairs the old LEX_C=1e4 scalar misordered.
# --------------------------------------------------------------------------- #
class TestLexOrder(unittest.TestCase):
    def test_prefers_better_T1(self):
        """A=(5.00,10,10) beats B=(5.01,5,5) on the primary objective."""
        T = np.array([[5.00, 10.0, 10.0],
                      [5.01,  5.0,  5.0]], dtype=np.float64)
        self.assertEqual(lex_argbest(T), 0)
        self.assertLess(lex_key(T[0]), lex_key(T[1]))

    def test_ties_break_on_lower_levels(self):
        """Equal T1 -> T2 decides; equal T1,T2 -> T3 decides."""
        T = np.array([[5.0, 9.0, 50.0],
                      [5.0, 9.5,  1.0],
                      [5.0, 9.0, 49.0]], dtype=np.float64)
        self.assertEqual(lex_sort(T).tolist(), [2, 0, 1])

    def test_generalizes_over_P(self):
        """Comparison is componentwise over however many levels are present,
        not hardcoded P=3."""
        T = np.array([[1.0, 2.0, 3.0, 10.0],
                      [1.0, 2.0, 3.0,  9.0]], dtype=np.float64)
        self.assertEqual(lex_argbest(T), 1)

    def test_near_tie_immune_to_scalar_folding(self):
        """Real pair from an EA run on data/test_bm/small.npz[1] (variant P):
        A's T1 is strictly better by ~6e-8, so the old LEX_C=1e4 Horner
        scalar (T1 weight 1e8) let B's ~2 s T2 advantage outvote it — a
        silent lex misordering. Exact comparison must pick A."""
        A = np.array([6.27379846572876, 16.42958915233612, 25.919182538986206])
        B = np.array([6.2737985253334045, 14.437795132398605, 21.55321180820465])
        w = 1e4 ** np.arange(2, -1, -1, dtype=np.float64)
        self.assertLess(B @ w, A @ w)                       # scalar misorders
        self.assertEqual(lex_argbest(np.stack([A, B])), 0)  # exact lex does not
        self.assertLess(lex_key(A), lex_key(B))

    def test_magnitude_robust_at_higher_P(self):
        """For P=6 the Horner scalar needs C**5 = 1e20 of float64 dynamic
        range, so a last-level difference is not even representable — the
        scalars collapse to exact equality. Componentwise comparison is exact
        for any P and any magnitude."""
        P = 6
        A = np.array([1e2] * (P - 1) + [1.0])
        B = np.array([1e2] * (P - 1) + [1.0 + 1e-9])
        w = 1e4 ** np.arange(P - 1, -1, -1, dtype=np.float64)
        self.assertEqual(A @ w, B @ w)                      # scalar collapses
        self.assertEqual(lex_argbest(np.stack([A, B])), 0)
        self.assertLess(lex_key(A), lex_key(B))


# --------------------------------------------------------------------------- #
# Step 2. ILSHCARP end-to-end
# --------------------------------------------------------------------------- #
class TestILS(unittest.TestCase):
    def test_ils_improves_over_construction(self):
        al = make_instance(nv=3, seed=5)

        # Replicate ILS's own initialization (seeding np.random the same way
        # al(seed=123) does replays the identical sample stream): 5 constructive
        # samples -> local search -> best. ILS's best-ever tracking starts from
        # exactly this solution, so its final answer must match or beat it.
        # (An *external* construction sample is NOT a valid baseline: ILS never
        # sees it, and exact lex comparison has no weighted-sum slack.)
        np.random.seed(123)
        samples = [al.get_once(al.M, al.clss) for _ in range(5)]
        samples = [s for s in samples if s is not None]
        self.assertTrue(samples)
        tours_batch = ls(al.vars, variant='P', actions=samples)
        action_batch = deserialize_tours_batch(tours_batch, al.nseq)
        base_T = lex_key(al.get_best(action_batch)[0])

        T = al(max_iter=50, variant='P', num_init_sample=5, seed=123)
        # get_Ts returns shape (1, P)
        self.assertEqual(T.shape, (1, 3))
        # Both sides are evaluator T-vectors; compare them the way the solver
        # itself does — exact lex (tuple) order, lower is better.
        ils_T = lex_key(al.calc_Ts(al._last_best_action[None])[0])

        # ILS result must be at least as good as its own initialization
        self.assertLessEqual(ils_T, base_T)

    def test_ils_sa_mode_runs(self):
        al = make_instance(nv=3, seed=7)
        al.accept_mode = 'sa'
        T = al(max_iter=30, variant='U', num_init_sample=3, seed=7)
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T >= 0))


# --------------------------------------------------------------------------- #
# 2b. ls_lex_guarded — regression for the ILS-U T1 blow-up. The inter/intra
#     ``ls`` minimises total deadheading, NOT the lex (T1,T2,T3) objective;
#     under variant U its unrestricted any-class moves freely drag class-1
#     arcs deep into routes (less deadhead, catastrophically worse T1).
#     Measured on osm_test_small: unguarded, the first ls pass alone pushed
#     T1 +40..80% and the ILS-U benchmark came out 39% worse than ILS-P on
#     T1. ``ls_lex_guarded`` must never return a lex-worse action than its
#     input, and under U keeps a class-restricted P arm (every P-move is
#     U-feasible) so U retains at least P's local-search power.
# --------------------------------------------------------------------------- #
def make_adversarial_instance():
    """Depot + 3 arcs where the deadhead optimum has strictly worse T1.

    Arcs 1 (=a) and 2 (=b) are class 1, arc 3 (=c) is class 2; M=1; all
    services are 1. Total deadhead of order [a,b,c] is 1+1+10+1 = 13, of
    [b,a,c] it is 2+2+1+1 = 6 (the global minimum) — so the deadhead-only
    local search flips the class-1 pair. But the class-1 prefix gets slower:
    T1([a,b,·]) = 1+1+1+1 = 4 vs T1([b,a,·]) = 2+1+2+1 = 6. The flip is a
    WITHIN-class reorder, so both U decode arms (order-preserving segment and
    class-sorted _build_P) inherit the damage: the T-vector strictly
    lex-worsens under both variants' scoring. The P-restricted ls finds the same flip
    (same-class swaps are P-legal), so only the pre-LS fallback of
    ``ls_lex_guarded`` protects T1 here.
    """
    n = 4
    adj = np.full((n, n), 10.0)
    np.fill_diagonal(adj, 0.0)
    adj[0, 1] = 1.0   # depot -> a
    adj[1, 2] = 1.0   # a -> b
    adj[0, 2] = 2.0   # depot -> b
    adj[2, 1] = 2.0   # b -> a
    adj[1, 3] = 1.0   # a -> c
    adj[3, 0] = 1.0   # c -> depot

    al = ILSHCARP(strength=1)
    al.dms = adj
    al.P = [1, 2, 3]
    al.M = [0]                                     # single vehicle
    al.demands = np.array([0.0, 0.1, 0.1, 0.1], dtype=np.float32)
    al.clss = np.array([0, 1, 1, 2], dtype=np.int32)
    al.s = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    al.d = np.zeros(n, dtype=np.float64)
    al.nv = 1
    al.nseq = n
    al.max_len = al.nseq + (al.nv - 2)
    al.vars = {
        'adj': al.dms,
        'service_time': al.s,
        'clss': al.clss,
        'demand': al.demands,
        'nv': al.nv,
    }
    al.has_instance = True
    return al


class TestLsLexGuarded(unittest.TestCase):
    def test_unguarded_u_ls_lex_worsens(self):
        """Document the failure mode: plain ls(variant='U') on the adversarial
        instance strictly lex-worsens the T-vector (it only sees deadhead)."""
        al = make_adversarial_instance()
        al.variant = 'U'
        pre = np.array([1, 2, 3], dtype=np.int32)
        tours = ls(al.vars, variant='U', actions=[pre])
        post = deserialize_tours_batch(tours, al.nseq)[0]
        self.assertGreater(lex_key(al.calc_Ts(post[None])[0]),
                           lex_key(al.calc_Ts(pre[None])[0]))

    def test_guard_never_lex_worsens_adversarial(self):
        al = make_adversarial_instance()
        al.variant = 'U'
        pre = np.array([1, 2, 3], dtype=np.int32)
        acts, Ts = al.ls_lex_guarded([pre])
        self.assertLessEqual(lex_key(Ts[0]),
                             lex_key(al.calc_Ts(pre[None])[0]))
        # T1 must stay at its optimum (a then b: 1+1 + 1+1 = 4)
        T = evaluate_batch(al.vars, actions=[acts[0]], variant='U')[0]
        self.assertAlmostEqual(float(T[0]), 4.0, places=9)

    def test_guard_never_lex_worsens_random(self):
        """Property: for every sample and both variants, the guarded LS
        T-vector is lex better-or-equal to the pre-LS T-vector."""
        for variant in ('P', 'U'):
            al = make_instance(nv=3, seed=11)
            al.variant = variant
            np.random.seed(42)
            samples = [al.get_once(al.M, al.clss) for _ in range(8)]
            samples = [s for s in samples if s is not None]
            self.assertTrue(samples)
            pre_Ts = al.calc_Ts(samples)
            _, Ts = al.ls_lex_guarded(samples)
            for post, pre in zip(Ts, pre_Ts):
                self.assertLessEqual(lex_key(post), lex_key(pre),
                                     msg=f"variant {variant}: {post} > {pre}")

    def test_ils_u_end_to_end_keeps_t1(self):
        """Full ILS under U on the adversarial instance must report the
        T1-optimal solution (unguarded, ls flips the class-1 pair for less
        deadhead and T1 comes out 6)."""
        al = make_adversarial_instance()
        T = al(max_iter=30, variant='U', num_init_sample=3, seed=3)
        self.assertAlmostEqual(float(T[0, 0]), 4.0, places=9)


# --------------------------------------------------------------------------- #
# Step 3. variant propagation to the evaluator (regression: evaluate_batch used
# to silently default to 'P', so ILS/EA/ACO runs with variant='U' were scored
# through the P route-builder and any cross-class interleaving was destroyed
# before it was ever measured)
# --------------------------------------------------------------------------- #
class TestMetaVariantPropagation(unittest.TestCase):
    def _spied_run(self, variant, seed=11):
        from solvers.evaluator import evaluate_batch
        al = make_instance(nv=3, seed=9)
        seen = []

        def spy(vars, actions, pos_val=(1, 2, 3), variant='P'):
            # records the variant each get_Ts call actually receives; a call
            # site that forgot the kwarg shows up here as the 'P' default.
            seen.append(variant)
            return evaluate_batch(vars, actions, pos_val=pos_val,
                                  variant=variant)

        with mock.patch('solvers.meta.get_Ts', side_effect=spy):
            T = al(max_iter=5, variant=variant, num_init_sample=3, seed=seed)
        return al, T, seen

    def test_ils_scores_with_U_evaluator(self):
        al, T, seen = self._spied_run('U')
        self.assertEqual(al.variant, 'U')
        self.assertTrue(seen, "get_Ts was never called")
        self.assertTrue(all(v == 'U' for v in seen),
                        f"some get_Ts call fell back to P: {set(seen)}")
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(T)))

    def test_ils_scores_with_P_evaluator(self):
        al, _, seen = self._spied_run('P')
        self.assertEqual(al.variant, 'P')
        self.assertTrue(seen and all(v == 'P' for v in seen))


if __name__ == '__main__':
    unittest.main()
