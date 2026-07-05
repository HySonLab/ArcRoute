import os
import sys
import unittest
from collections import Counter
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.ops import gen_tours, convert_prob, centered_lex_rank, deserialize_tours_batch
from utils.local_search import ls
from solvers.evaluator import evaluate_batch as get_Ts
from solvers.meta import EAHCARP


def make_symmetric_adj(n, seed=0):
    """Random symmetric arc-to-arc matrix with zero diagonal (n includes depot)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float64)


def arc_multiset(action):
    """Counter of all non-depot arcs in a flat action (or list of routes)."""
    arr = np.asarray(action).ravel()
    return Counter(int(a) for a in arr if int(a) != 0)


def make_instance(nv=3, seed=5, **ea_kwargs):
    """Build an EAHCARP with attributes set directly (no .npz needed).

    6 arcs + depot; classes 1,1,2,2,3,3; equal demands (0.3 each).
    """
    n = 7  # 6 arcs + depot
    adj = make_symmetric_adj(n, seed=seed)
    clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    demands = np.full(n, 0.3, dtype=np.float64)
    demands[0] = 0.0
    s = np.zeros(n, dtype=np.float64)   # service times
    d = np.zeros(n, dtype=np.float64)   # traversal

    al = EAHCARP(**ea_kwargs)
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


# the full arc set every individual must cover (indices 1..6, once each)
EXPECTED_ARCS = Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})


# --------------------------------------------------------------------------- #
# TestInitPopulation
# --------------------------------------------------------------------------- #
class TestInitPopulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()

    def test_init_population_size(self):
        self.assertGreater(len(self.pop), 0)
        self.assertLessEqual(len(self.pop), self.al.n_population)

    def test_init_population_no_none(self):
        for ind in self.pop:
            self.assertIsNotNone(ind)

    def test_init_population_all_arcs_present(self):
        for ind in self.pop:
            self.assertEqual(arc_multiset(ind), EXPECTED_ARCS)

    def test_init_population_feasible(self):
        for ind in self.pop:
            self.assertTrue(self.al.is_valid(ind))


# --------------------------------------------------------------------------- #
# TestIsValid (BaseHCARP)
# --------------------------------------------------------------------------- #
class TestIsValid(unittest.TestCase):
    def setUp(self):
        self.al = make_instance(nv=3, seed=5)

    def test_is_valid_feasible(self):
        # two routes, each load 3 * 0.3 = 0.9 <= 1.0
        action = np.array([1, 2, 3, 0, 4, 5, 6], dtype=np.int32)
        self.assertTrue(self.al.is_valid(action))

    def test_is_valid_overcapacity(self):
        # first route has 4 arcs => load 1.2 > 1.0
        action = np.array([1, 2, 3, 4, 0, 5, 6], dtype=np.int32)
        self.assertFalse(self.al.is_valid(action))

    def test_is_valid_ragged_routes(self):
        # routes of different lengths must not crash (regression for the
        # ragged fancy-index bug). 3 / 2 / 1 arcs, all feasible.
        action = np.array([1, 2, 3, 0, 4, 5, 0, 6], dtype=np.int32)
        result = self.al.is_valid(action)
        self.assertIsInstance(result, bool)
        self.assertTrue(result)


# --------------------------------------------------------------------------- #
# TestCrossOver
# --------------------------------------------------------------------------- #
class TestCrossOver(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.p1, self.p2 = self.pop[0], self.pop[1]

    def test_crossover_conserves_arcs(self):
        for _ in range(10):
            child = self.al.cross_over(self.p1, self.p2)
            self.assertEqual(arc_multiset(child), EXPECTED_ARCS)

    def test_crossover_feasible(self):
        for _ in range(10):
            child = self.al.cross_over(self.p1, self.p2)
            self.assertTrue(self.al.is_valid(child))

    def test_crossover_fallback_to_p1(self):
        with mock.patch.object(self.al, '_cross_over', return_value=None):
            child = self.al.cross_over(self.p1, self.p2, attemp=5)
        self.assertTrue(np.array_equal(child, self.p1))

    def test_crossover_nv2_edge_case(self):
        np.random.seed(2)
        al = make_instance(nv=2, seed=5, n_population=20)
        pop = al.init_population()
        child = al.cross_over(pop[0], pop[1])
        self.assertEqual(arc_multiset(child), EXPECTED_ARCS)
        self.assertTrue(al.is_valid(child))


# --------------------------------------------------------------------------- #
# TestMutate
# --------------------------------------------------------------------------- #
class TestMutate(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.ind = self.al.init_population()[0]

    def test_mutate_conserves_arcs(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertEqual(arc_multiset(out), EXPECTED_ARCS)

    def test_mutate_feasible(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertTrue(self.al.is_valid(out))

    def test_mutate_returns_correct_length(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertEqual(len(out), self.al.max_len)


# --------------------------------------------------------------------------- #
# TestOperate
# --------------------------------------------------------------------------- #
class TestOperate(unittest.TestCase):
    def setUp(self):
        np.random.seed(4)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.prob = np.full(len(self.pop), 1.0 / len(self.pop))

    def test_operate_returns_two_children(self):
        out = self.al.operate(self.prob, self.pop, variant='P')
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_operate_children_conserve_arcs(self):
        for _ in range(10):
            c1, c2 = self.al.operate(self.prob, self.pop, variant='P')
            self.assertEqual(arc_multiset(c1), EXPECTED_ARCS)
            self.assertEqual(arc_multiset(c2), EXPECTED_ARCS)

    def test_operate_always_defined(self):
        # Force every combination of the crossover / mutation branches and make
        # sure both children are always defined (no UnboundLocalError).
        for r in (0.0, 1.0):
            with mock.patch('solvers.meta.random', return_value=r):
                c1, c2 = self.al.operate(self.prob, self.pop, variant='P')
                self.assertIsNotNone(c1)
                self.assertIsNotNone(c2)


# --------------------------------------------------------------------------- #
# TestGetParent
# --------------------------------------------------------------------------- #
class TestGetParent(unittest.TestCase):
    def setUp(self):
        np.random.seed(5)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.prob = np.full(len(self.pop), 1.0 / len(self.pop))

    def _matching_indices(self, parent):
        return [i for i, row in enumerate(self.pop)
                if np.array_equal(np.asarray(row), np.asarray(parent))]

    def test_get_parent_returns_valid_index(self):
        parent = self.al.get_parent(self.pop, self.prob)
        self.assertTrue(len(self._matching_indices(parent)) > 0,
                        msg="returned parent is not a member of the population")

    def test_get_parent_distribution(self):
        seen = set()
        for _ in range(1000):
            parent = self.al.get_parent(self.pop, self.prob)
            seen.update(self._matching_indices(parent))
        self.assertEqual(seen, set(range(len(self.pop))))


# --------------------------------------------------------------------------- #
# TestSelectionProb — roulette-wheel selection must stay non-degenerate
# --------------------------------------------------------------------------- #
class TestSelectionProb(unittest.TestCase):
    """EA's parent-selection probabilities are softmax over centered_lex_rank
    (bounded [-1, +1]) rather than over the exact-lex calc_obj scalar, whose
    LEX_C-scale dynamic range would collapse softmax to near-one-hot."""

    def setUp(self):
        np.random.seed(5)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()

    def test_selection_prob_non_degenerate(self):
        # exactly the expression EAHCARP.__call__ uses for `prob`
        T = get_Ts(self.al.vars, actions=self.pop)
        prob = convert_prob(centered_lex_rank(T, gamma=0.5))
        self.assertAlmostEqual(float(prob.sum()), 1.0, places=9)
        self.assertLess(float(prob.max()), 0.9)
        self.assertGreater(float(prob.min()), 0.0)

    def test_selection_prob_raw_lex_obj_would_degenerate(self):
        # contrast: softmax over the exact-lex objective IS near-one-hot for a
        # population whose individuals meaningfully differ (deterministic
        # hand-built T spread) — the reason EA must rank before softmaxing.
        T = np.array([[5.0, 10.0, 10.0],
                      [5.2,  9.0,  9.0],
                      [6.0,  8.0,  8.0],
                      [7.0,  7.0,  7.0]], dtype=np.float64)
        with mock.patch('solvers.meta.get_Ts', return_value=T):
            obj = self.al.calc_obj(np.zeros((4, 3), dtype=np.int32))
        prob_raw = convert_prob(obj)
        self.assertGreater(float(prob_raw.max()), 0.99)
        # ...while the rank-based probability stays smooth for the same T
        prob_rank = convert_prob(centered_lex_rank(T, gamma=0.5))
        self.assertLess(float(prob_rank.max()), 0.9)


# --------------------------------------------------------------------------- #
# TestMutateLexGuarded — regression for the EA-U T1 blow-up (same bug class as
# tests/test_ils.py::TestLsLexGuarded): ``ls`` minimises total deadheading,
# NOT the lex (T1,T2,T3) objective, and under variant U its unrestricted
# any-class moves trade "class-1 served early" for less deadhead. mutate()
# must therefore go through BaseHCARP.ls_lex_guarded, never raw ``ls``.
# --------------------------------------------------------------------------- #
def make_adversarial_instance():
    """Depot + 3 arcs where the deadhead optimum has strictly worse T1.

    Adapted verbatim from tests/test_ils.py::make_adversarial_instance, built
    as an EAHCARP: arcs 1,2 are class 1, arc 3 is class 2; M=1; all services
    are 1. Total deadhead of order [1,2,3] is 13, of [2,1,3] it is 6 (the
    global minimum) — so the deadhead-only local search flips the class-1
    pair, but the class-1 prefix gets slower: T1 goes 4 -> 6. The P-restricted
    ls finds the same flip (same-class swaps are P-legal), so only the pre-LS
    fallback of ls_lex_guarded protects T1 here.
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

    al = EAHCARP(n_population=4)
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


class TestMutateLexGuarded(unittest.TestCase):
    def test_unguarded_u_ls_lex_worsens(self):
        """Document the failure mode: plain ls(variant='U') on the adversarial
        instance strictly lex-worsens calc_obj (it only sees deadhead) — this
        is exactly what the old mutate() did."""
        al = make_adversarial_instance()
        al.variant = 'U'
        pre = np.array([1, 2, 3], dtype=np.int32)
        tours = ls(al.vars, variant='U', actions=[pre])
        post = deserialize_tours_batch(tours, al.nseq)[0]
        self.assertLess(float(al.calc_obj(post[None])[0]),
                        float(al.calc_obj(pre[None])[0]))

    def test_mutate_u_keeps_t1_adversarial(self):
        al = make_adversarial_instance()
        al.variant = 'U'
        pre = np.array([1, 2, 3], dtype=np.int32)
        out = al.mutate(pre, variant='U')
        self.assertGreaterEqual(float(al.calc_obj(out[None])[0]),
                                float(al.calc_obj(pre[None])[0]) - 1e-9)
        # T1 must stay at its optimum (a then b: 1+1 + 1+1 = 4)
        T = get_Ts(al.vars, actions=out[None], variant='U')[0]
        self.assertAlmostEqual(float(T[0]), 4.0, places=9)

    def test_mutate_never_lex_worsens_random(self):
        """Property: for every individual and both variants, mutate's output
        objective is >= the input's (the ls_lex_guarded guarantee)."""
        for variant in ('P', 'U'):
            np.random.seed(42)
            al = make_instance(nv=3, seed=11, n_population=8)
            al.variant = variant
            pop = al.init_population()
            pre = al.calc_obj(pop)
            for ind, p in zip(pop, pre):
                out = al.mutate(ind, variant=variant)
                self.assertGreaterEqual(
                    float(al.calc_obj(out[None])[0]), float(p) - 1e-9,
                    msg=f"variant {variant}: mutate lex-worsened an individual")

    def test_mutate_variant_mismatch_asserts(self):
        """mutate scores through self.variant (set by __call__); a caller
        passing a different variant must fail loudly, not score silently
        with the wrong decoder."""
        al = make_instance(nv=3, seed=5, n_population=4)
        al.variant = 'P'
        with self.assertRaises(AssertionError):
            al.mutate(np.array([1, 2, 3, 0, 4, 5, 6], dtype=np.int32),
                      variant='U')


# --------------------------------------------------------------------------- #
# TestGetOncePrecheck — bin-packing lower-bound shortcut in BaseHCARP.get_once.
# When the eligible arcs' total demand exceeds the fleet's total capacity
# (1.0/vehicle), construction is impossible; get_once must return None
# immediately WITHOUT consuming np.random draws on doomed attempts (this
# futile work dominated EAHCARP.cross_over on tight instances: ~96% of
# get_once calls / ~88% of EA wall time on osm_test_small).
# --------------------------------------------------------------------------- #
class TestGetOncePrecheck(unittest.TestCase):
    def setUp(self):
        np.random.seed(21)
        self.al = make_instance(nv=3, seed=5, n_population=4)

    def test_infeasible_returns_none_without_rng(self):
        # 6 arcs x 0.6 demand = 3.6 > 1 vehicle capacity -> deterministic fail
        self.al.demands = np.full(7, 0.6, dtype=np.float32)
        self.al.demands[0] = 0.0
        state = np.random.get_state()
        out = self.al.get_once([0], self.al.clss)
        self.assertIsNone(out)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]),
                        msg="doomed get_once consumed np.random draws")

    def test_feasible_still_constructs(self):
        # total demand 6 * 0.3 = 1.8 <= 3 vehicles -> precheck must not fire
        out = self.al.get_once(self.al.M, self.al.clss)
        self.assertIsNotNone(out)
        self.assertEqual(arc_multiset(out), EXPECTED_ARCS)
        self.assertTrue(self.al.is_valid(out))

    def test_zeroed_classes_excluded_from_bound(self):
        # zero out enough arcs that the remainder fits one vehicle: the bound
        # must only count still-eligible (clss in P) arcs.
        clss = self.al.clss.copy()
        clss[[1, 2, 3, 4]] = 0                       # only arcs 5, 6 left (0.6)
        out = self.al.get_once([0], clss)
        self.assertIsNotNone(out)
        self.assertEqual(arc_multiset(out), Counter({5: 1, 6: 1}))


# --------------------------------------------------------------------------- #
# TestBatchedMutationEquivalence — EAHCARP.__call__ guards a whole
# generation's flagged children in ONE ls_lex_guarded call. That is only
# byte-identical to the historical per-child mutate() loop if ls_lex_guarded
# treats every sample independently: same tours, same objectives, same
# per-sample argmax, regardless of batch composition.
# --------------------------------------------------------------------------- #
class TestBatchedMutationEquivalence(unittest.TestCase):
    def test_batched_guard_equals_per_child_mutate(self):
        for variant in ('P', 'U'):
            np.random.seed(31)
            al = make_instance(nv=3, seed=9, n_population=8)
            al.variant = variant
            pop = al.init_population()
            solo = [al.mutate(ind, variant=variant) for ind in pop]
            batched, _ = al.ls_lex_guarded(list(pop))
            for s, b in zip(solo, batched):
                self.assertTrue(np.array_equal(s, b),
                                msg=f"variant {variant}: batched guard diverged "
                                    f"from per-child mutate")


# --------------------------------------------------------------------------- #
# TestEAEndToEnd
# --------------------------------------------------------------------------- #
class TestEAEndToEnd(unittest.TestCase):
    def test_ea_improves_over_init(self):
        np.random.seed(7)
        al = make_instance(nv=3, seed=5, n_population=20)

        pop = al.init_population()
        rand_ind = pop[np.random.randint(len(pop))]
        base_obj = float(al.calc_obj(rand_ind[None])[0])

        T = al(n_epoch=10, variant='P')
        self.assertEqual(T.shape, (1, 3))
        # Route both sides through calc_obj so the comparison is decoupled
        # from calc_obj's internal weighting formula.
        ea_obj = float(al.calc_obj(al._last_best_action[None])[0])

        self.assertGreaterEqual(ea_obj, base_obj - 1e-6)

    def test_ea_result_feasible(self):
        np.random.seed(8)
        al = make_instance(nv=3, seed=5, n_population=20)
        T = al(n_epoch=10, variant='P')
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T > 0),
                        msg=f"expected all T_k > 0, got {T}")

    def test_ea_last_best_action_set(self):
        np.random.seed(9)
        al = make_instance(nv=3, seed=5, n_population=20)
        al(n_epoch=10, variant='P')
        self.assertTrue(hasattr(al, '_last_best_action'))
        self.assertIsNotNone(al._last_best_action)
        self.assertEqual(arc_multiset(al._last_best_action), EXPECTED_ARCS)


if __name__ == '__main__':
    unittest.main()
