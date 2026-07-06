"""Tests for the RL hybrid solver (`solvers.rl`) and the exact T1 oracle
head (`solvers.oracle`, P2). Also covers the P2->single-pool unification
(docs/grpo.md §17/§18): the old two-arm design (separate pool-arm and
oracle-head-arm decodes joined by `oracle_union_pick`) was replaced by one
batched decode (`_decode_pool`) over a combined free+forced-row pool, with
one final selection rule (`final_band_pick`).

Covers:
  1. TestT1OracleArgmin      — oracle T1 matches brute force; `parts` are
     internally consistent (coverage, capacity, completion formula);
     tiebreak preserves T1 and never worsens the secondary; guards
     (max_n1, capacity-infeasible, no class-1 arcs) return "not applicable".
  2. TestFinalBandPick       — the single post-unification selection rule
     (replaces the old shortlist_completions/t2_band_pick/oracle_union_pick
     split) reduces to argmin-T1 absent a genuine tie, and to a (T2, T3)
     choice within a T1 tie-band.
  3. TestForcedDecode        — forcing the oracle's `parts` through
     `_decode_pool`'s merged batch reproduces the oracle T1 (fp32 tolerance).
  4. TestOracleHeadRegression — oracle_head=False (default) still delegates
     to the plain policy rollout unchanged (`_decode_pool`'s byte-identical
     fallback when no prefix is forced); the guard fallback does too;
     oracle_head=True returns a T1-optimal answer end-to-end. (The final
     SELECTION step is no longer asserted byte-identical to the frozen pre-
     unification pipeline -- it intentionally now uses `final_band_pick`
     instead of strict-lex argmin; see docs/grpo.md §17/§18 for the
     empirical validation that this doesn't regress quality.)

Run: uv run python -m unittest tests.test_rl -v
"""
import os
import tempfile
import unittest
from itertools import permutations, product
from unittest import mock

import numpy as np
import torch


# --------------------------------------------------------------------- helpers
def _completion(adj, svc, seq):
    """Prefix completion: svc[0] + adj[0,a1] + svc[a1] + adj[a1,a2] + ..."""
    t = float(svc[0]) + float(adj[0, seq[0]]) + float(svc[seq[0]])
    for a, b in zip(seq, seq[1:]):
        t += float(adj[a, b]) + float(svc[b])
    return t


def _brute_T1(adj, svc, clss, demands, M, cap=1.0):
    """Exact T1 by full enumeration (assignments x permutations)."""
    c1 = np.where(clss == 1)[0]
    best = float("inf")
    for assign in product(range(M), repeat=len(c1)):
        groups = [[int(c1[i]) for i in range(len(c1)) if assign[i] == v]
                  for v in range(M)]
        if any(demands[g].sum() > cap + 1e-9 for g in groups if g):
            continue
        worst = 0.0
        for g in groups:
            if not g:
                continue
            worst = max(worst, min(_completion(adj, svc, list(p))
                                   for p in permutations(g)))
        best = min(best, worst)
    return best


def _hand_instance(seed=0, M=2):
    """Small deterministic instance in the `iter_instances` data-tuple format.

    6 required arcs + depot; class-1 demand sum > capacity so the optimal
    class-1 partition is forced to use both vehicles (nontrivial oracle)."""
    rng = np.random.RandomState(seed)
    N1 = 7
    adj = rng.uniform(0.1, 1.0, size=(N1, N1)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    clss = np.array([0, 1, 1, 1, 2, 2, 3], dtype=np.int32)
    demands = np.array([0, .4, .35, .3, .3, .3, .2], dtype=np.float32)
    svc = np.array([0, .1, .2, .15, .1, .1, .2], dtype=np.float32)
    d = (svc / 2).astype(np.float32)
    veh = list(range(M))
    edge_indxs = np.zeros((N1, 2), dtype=np.int32)
    return adj, [1, 2, 3], veh, demands, clss, svc, d, edge_indxs


_CKPT_CACHE = {}


def _fake_ckpt():
    """Serialize a small random policy as a Lightning-style checkpoint so the
    real RLHCARP loading path is exercised without a trained model."""
    if "path" not in _CKPT_CACHE:
        from policy.policy import AttentionModelPolicy
        torch.manual_seed(0)
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1,
                                      num_heads=4)
        ckpt = {
            "state_dict": {"policy." + k: v for k, v in policy.state_dict().items()},
            "hyper_parameters": {"embed_dim": 32, "num_encoder_layers": 1,
                                 "num_heads": 4},
        }
        fd, path = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        torch.save(ckpt, path)
        _CKPT_CACHE["path"] = path
    return _CKPT_CACHE["path"]


def _make_hyb(**kw):
    from solvers.rl import RLHybHCARP
    al = RLHybHCARP(_fake_ckpt(), "P", device="cpu", **kw)
    al._load_data(*_hand_instance())
    return al


# ----------------------------------------------------------------------- 1
class TestT1OracleArgmin(unittest.TestCase):
    def setUp(self):
        adj, _, veh, demands, clss, svc, _, _ = _hand_instance()
        self.adj = adj.astype(np.float64)
        self.svc = svc.astype(np.float64)
        self.clss = clss.astype(np.int64)
        self.dem = demands.astype(np.float64)
        self.M = len(veh)

    def test_matches_brute_force_and_parts_consistent(self):
        from solvers.oracle import t1_oracle_argmin
        for tiebreak in (False, True):
            res = t1_oracle_argmin(self.adj, self.svc, self.clss, self.dem,
                                   self.M, tiebreak=tiebreak)
            brute = _brute_T1(self.adj, self.svc, self.clss, self.dem, self.M)
            self.assertAlmostEqual(res["T1"], brute, places=9)

            parts = res["parts"]
            self.assertEqual(len(parts), self.M)
            # coverage: class-1 arcs exactly once, nothing else
            served = [a for p in parts for a in p]
            self.assertEqual(sorted(served),
                             sorted(np.where(self.clss == 1)[0].tolist()))
            # capacity feasibility per part
            for p in parts:
                if p:
                    self.assertLessEqual(self.dem[p].sum(), 1.0 + 1e-9)
            # T1 == max per-part completion under the stated formula
            comp = max(_completion(self.adj, self.svc, p) for p in parts if p)
            self.assertAlmostEqual(res["T1"], comp, places=9)

    def test_tiebreak_never_worsens_secondary(self):
        from solvers.oracle import t1_oracle_argmin
        res = t1_oracle_argmin(self.adj, self.svc, self.clss, self.dem,
                               self.M, tiebreak=True)
        self.assertLessEqual(res["secondary"], res["sec_plain"] + 1e-9)
        self.assertAlmostEqual(res["tie_gain"],
                               res["sec_plain"] - res["secondary"], places=9)

    def test_max_n1_guard(self):
        from solvers.oracle import t1_oracle_argmin
        res = t1_oracle_argmin(self.adj, self.svc, self.clss, self.dem,
                               self.M, max_n1=2)          # n1 = 3 > 2
        self.assertEqual(res["T1"], float("inf"))
        self.assertIsNone(res["parts"])

    def test_capacity_infeasible_returns_inf(self):
        from solvers.oracle import t1_oracle_argmin
        dem = self.dem.copy()
        dem[1] = 1.5                                       # class-1 arc > cap
        res = t1_oracle_argmin(self.adj, self.svc, self.clss, dem, self.M)
        self.assertEqual(res["T1"], float("inf"))
        self.assertIsNone(res["parts"])

    def test_no_class1_arcs(self):
        from solvers.oracle import t1_oracle_argmin
        clss = self.clss.copy()
        clss[clss == 1] = 2
        res = t1_oracle_argmin(self.adj, self.svc, clss, self.dem, self.M)
        self.assertEqual(res["T1"], 0.0)
        self.assertEqual(res["parts"], [[] for _ in range(self.M)])

    def test_forced_flat_actions(self):
        from solvers.oracle import forced_flat_actions
        self.assertEqual(forced_flat_actions([[2, 1], [], [3]], N1=7),
                         [2, 1, 2 * 7 + 3])


# ------------------------------------------------------------------- 1b
class TestT1HeuristicArgmin(unittest.TestCase):
    """`t1_heuristic_argmin` -- the large-n1 fallback for `t1_oracle_argmin`
    (P2's medium/large-curriculum extension). Never claims exactness; these
    tests check it's never *below* the exact optimum, produces internally
    consistent (coverage/capacity/completion-formula) parts, and degrades
    to `inf`/`None` gracefully exactly like the oracle does."""

    def setUp(self):
        adj, _, veh, demands, clss, svc, _, _ = _hand_instance()
        self.adj = adj.astype(np.float64)
        self.svc = svc.astype(np.float64)
        self.clss = clss.astype(np.int64)
        self.dem = demands.astype(np.float64)
        self.M = len(veh)

    def test_matches_exact_and_parts_consistent(self):
        from solvers.oracle import t1_heuristic_argmin, t1_oracle_argmin
        exact = t1_oracle_argmin(self.adj, self.svc, self.clss, self.dem, self.M)
        res = t1_heuristic_argmin(self.adj, self.svc, self.clss, self.dem, self.M,
                                  time_budget=0.2, seed=0)
        # never beats the exact optimum; on this tiny instance should match it.
        self.assertGreaterEqual(res["T1"], exact["T1"] - 1e-6)
        self.assertAlmostEqual(res["T1"], exact["T1"], places=6)

        parts = res["parts"]
        self.assertEqual(len(parts), self.M)
        served = [a for p in parts for a in p]
        self.assertEqual(sorted(served),
                         sorted(np.where(self.clss == 1)[0].tolist()))
        for p in parts:
            if p:
                self.assertLessEqual(self.dem[p].sum(), 1.0 + 1e-9)
        comp = max(_completion(self.adj, self.svc, p) for p in parts if p)
        self.assertAlmostEqual(res["T1"], comp, places=6)

    def test_capacity_infeasible_returns_inf(self):
        from solvers.oracle import t1_heuristic_argmin
        dem = self.dem.copy()
        dem[1] = 1.5                                       # class-1 arc > cap
        res = t1_heuristic_argmin(self.adj, self.svc, self.clss, dem, self.M,
                                  time_budget=0.05)
        self.assertEqual(res["T1"], float("inf"))
        self.assertIsNone(res["parts"])

    def test_no_class1_arcs(self):
        from solvers.oracle import t1_heuristic_argmin
        clss = self.clss.copy()
        clss[clss == 1] = 2
        res = t1_heuristic_argmin(self.adj, self.svc, clss, self.dem, self.M,
                                  time_budget=0.05)
        self.assertEqual(res["T1"], 0.0)
        self.assertEqual(res["parts"], [[] for _ in range(self.M)])

    def test_cross_validates_on_larger_random_instance(self):
        """On a bigger synthetic instance (n1=12, still exact-DP-tractable),
        the heuristic should land within a small margin of the true optimum
        -- never below it."""
        from solvers.oracle import t1_heuristic_argmin, t1_oracle_argmin
        rng = np.random.RandomState(7)
        n1 = 12
        N1 = n1 + 5
        adj = rng.uniform(0.1, 1.0, size=(N1, N1))
        np.fill_diagonal(adj, 0.0)
        clss = np.array([0] + [1] * n1 + [2] * (N1 - 1 - n1))
        dem = rng.uniform(0.05, 0.15, size=N1)
        dem[0] = 0.0
        svc = rng.uniform(0.1, 0.3, size=N1)
        svc[0] = 0.0
        M = 3
        exact = t1_oracle_argmin(adj, svc, clss, dem, M, tiebreak=True)
        heur = t1_heuristic_argmin(adj, svc, clss, dem, M, time_budget=0.3, seed=1)
        self.assertTrue(np.isfinite(heur["T1"]))
        self.assertGreaterEqual(heur["T1"], exact["T1"] - 1e-6)
        rel_gap = (heur["T1"] - exact["T1"]) / exact["T1"]
        # Smoke-level regression bound, not a tight quality claim (single
        # random instance/seed) -- measured mean gap at this n1 range is
        # <0.1-2% depending on budget (docs/grpo.md), so 5% leaves headroom
        # against single-instance variance while still catching a real
        # regression (e.g. a construction/local-search bug reintroduced).
        self.assertLess(rel_gap, 0.05, "heuristic unexpectedly far from exact "
                                       "on an easily-DP-tractable n1=12 instance")

    def test_intra_cutoff_does_not_break_feasibility(self):
        """Both the 'full' (intra-route 2-opt/Or-opt) and lighter
        (relocate/swap-only) variants must still return capacity-feasible,
        fully-covering parts -- only measured quality differs (docs/grpo.md),
        not correctness."""
        from solvers.oracle import t1_heuristic_argmin
        for cutoff in (0, 1000):                # forces light vs full variant
            res = t1_heuristic_argmin(self.adj, self.svc, self.clss, self.dem,
                                      self.M, time_budget=0.1, intra_cutoff=cutoff)
            served = [a for p in res["parts"] for a in p]
            self.assertEqual(sorted(served),
                             sorted(np.where(self.clss == 1)[0].tolist()))
            for p in res["parts"]:
                if p:
                    self.assertLessEqual(self.dem[p].sum(), 1.0 + 1e-9)


# ----------------------------------------------------------------------- 2
class TestFinalBandPick(unittest.TestCase):
    """`final_band_pick` is the single post-unification selection rule,
    replacing the old `shortlist_completions` (forced-completion shortlist,
    now inlined as a plain lexsort in `RLHybHCARP.__call__`) and
    `t2_band_pick` + `oracle_union_pick` (arm-internal pick + two-arm union,
    now one T1-tie-band -> T2-eps-band -> min-T3 rule)."""

    def test_unique_t1_min_wins_regardless_of_own_t2_t3(self):
        from solvers.rl import final_band_pick
        Ts = np.array([[1.0, 9.0, 9.0],
                       [1.2, 0.0, 0.0],
                       [2.0, 0.0, 0.0]])
        self.assertEqual(final_band_pick(Ts, eps2=0.02), 0)

    def test_t1_tie_band_falls_back_to_t2_eps_band_then_t3(self):
        from solvers.rl import final_band_pick
        # same fixture shape as the old test_t2_band_pick/test_oracle_union_pick
        # T1-tied cases -- three rows share T1 (within tie_tol).
        Ts = np.array([[1.0, 0.500, 0.9],
                       [1.0, 0.505, 0.2],
                       [1.0, 0.600, 0.1],
                       [np.inf, np.inf, np.inf]])          # padding row ignored
        self.assertEqual(final_band_pick(Ts, 0.02), 1)     # band {0,1}, min T3
        self.assertEqual(final_band_pick(Ts, 0.0), 0)      # zero band -> argmin T2

    def test_t1_not_tied_prefers_strictly_better_t1(self):
        from solvers.rl import final_band_pick
        # head-style T1 strictly better / strictly worse (mirrors the old
        # oracle_union_pick's two-candidate cases, generalized to N rows).
        self.assertEqual(
            final_band_pick(np.array([[1.0, 2.0, 3.0], [0.9, 9.0, 9.0]]), 0.02), 1)
        self.assertEqual(
            final_band_pick(np.array([[1.0, 2.0, 3.0], [1.1, 0.0, 0.0]]), 0.02), 0)


# ----------------------------------------------------------------------- 3
class TestForcedDecode(unittest.TestCase):
    def test_forced_prefix_reproduces_oracle_T1(self):
        """Forcing the oracle's `parts` through `_decode_pool`'s merged batch
        (`num_free=0` -> every row is a forced row: 1 greedy + `oracle_r`
        sampled completions) yields decode-time T1 == oracle T1 (fp32
        arithmetic tolerance), for the greedy row and every sampled one."""
        from solvers.oracle import forced_flat_actions, t1_oracle_argmin
        torch.manual_seed(11)
        al = _make_hyb(shortlist=4, eps=0.01, oracle_r=7)
        td0 = al.td[0]
        N1 = td0["service_times"].shape[0]
        adj = td0["adj"].cpu().numpy().astype(np.float64)
        svc = td0["service_times"].cpu().numpy().astype(np.float64)
        clss = td0["clss"].cpu().numpy().astype(np.int64)
        dem = td0["demand"].cpu().numpy().astype(np.float64)
        M = int(td0["num_vehicle"])

        res = t1_oracle_argmin(adj, svc, clss, dem, M, tiebreak=True)
        self.assertTrue(np.isfinite(res["T1"]))
        forced = forced_flat_actions(res["parts"], N1)

        reward, actions, n_forced = al._decode_pool(0, forced)
        self.assertEqual(n_forced, 1 + al.oracle_r)
        self.assertTrue(np.allclose(reward[:, 0], res["T1"], rtol=1e-5),
                        f"decode T1 {reward[:, 0]} != oracle {res['T1']}")
        # forced opening is verbatim in every row's action sequence
        self.assertEqual(actions[:, :len(forced)].tolist(),
                         [forced] * actions.shape[0])


# ----------------------------------------------------------------------- 4
class TestOracleHeadRegression(unittest.TestCase):
    SEED = 123
    K = 16

    def test_default_off(self):
        al = _make_hyb()
        self.assertFalse(al.oracle_head)

    def test_oracle_code_not_reached_when_disabled(self):
        import solvers.oracle as oracle_mod
        al = _make_hyb(shortlist=4, eps=0.01)
        with mock.patch.object(oracle_mod, "t1_oracle_argmin",
                               side_effect=AssertionError("must not run")) as m1, \
             mock.patch.object(oracle_mod, "t1_heuristic_argmin",
                               side_effect=AssertionError("must not run")) as m2:
            torch.manual_seed(self.SEED)
            al(num_sample=self.K)
        m1.assert_not_called()
        m2.assert_not_called()

    def test_disabled_rollout_is_byte_identical_to_plain_policy(self):
        """Backward-compat guarantee for the default path, updated for the
        single-pool unification: with oracle_head=False, `_oracle_forced_prefix`
        returns None, so `_decode_pool` delegates straight to
        `AttentionModelPolicy.forward` -- the ROLLOUT (reward/actions) is
        byte-identical to a plain policy call on an identical RNG seed.

        (The final SELECTION among the shortlist now uses `final_band_pick`
        instead of strict-lex argmin -- see `TestFinalBandPick` for why that
        reduces to the same choice absent a genuine T1 tie, and
        docs/grpo.md §17/§18 for the end-to-end empirical validation that the
        full pipeline doesn't regress. This test only pins down the part of
        the guarantee that's still exact: the rollout itself.)"""
        from utils.ops import batchify
        al = _make_hyb(shortlist=4, eps=0.01, topk_t1=3)

        torch.manual_seed(self.SEED)
        td = batchify(al.td, self.K)
        with torch.inference_mode():
            ref_out = al.policy(td, env=al.env, decode_type="sampling")
        ref_reward = ref_out["reward"].cpu().numpy().astype(np.float64)
        ref_actions = ref_out["actions"].cpu().numpy().astype(np.int64)

        torch.manual_seed(self.SEED)
        got_reward, got_actions, n_forced = al._decode_pool(self.K, None)
        self.assertEqual(n_forced, 0)
        self.assertTrue(np.array_equal(ref_reward, got_reward))
        self.assertTrue(np.array_equal(ref_actions, got_actions))

    def test_guard_fallback_rollout_is_byte_identical_when_heuristic_disabled(self):
        """oracle_head=True, DP not applicable (max_n1 exceeded), heuristic
        explicitly disabled too -> `_oracle_forced_prefix` returns None (no
        solver applicable), so the rollout falls back to the plain policy
        call unchanged, exactly as when oracle_head=False (neither solver
        runs, no extra RNG consumed)."""
        from utils.ops import batchify
        al = _make_hyb(shortlist=4, eps=0.01, oracle_head=True, oracle_max_n1=2,
                       oracle_use_heuristic=False)
        td0 = al.td[0]
        N1 = td0["service_times"].shape[0]
        adj = td0["adj"].cpu().numpy().astype(np.float64)
        clss = td0["clss"].cpu().numpy().astype(np.int64)
        demands = td0["demand"].cpu().numpy().astype(np.float64)
        M = int(td0["num_vehicle"])
        forced = al._oracle_forced_prefix(td0, N1, adj, clss, demands, M)
        self.assertIsNone(forced)

        torch.manual_seed(self.SEED)
        td = batchify(al.td, self.K)
        with torch.inference_mode():
            ref_out = al.policy(td, env=al.env, decode_type="sampling")
        ref_reward = ref_out["reward"].cpu().numpy().astype(np.float64)

        torch.manual_seed(self.SEED)
        got_reward, got_actions, n_forced = al._decode_pool(self.K, forced)
        self.assertEqual(n_forced, 0)
        self.assertTrue(np.array_equal(ref_reward, got_reward))

    def test_heuristic_fallback_engages_above_max_n1(self):
        """oracle_head=True, DP not applicable (max_n1 exceeded), heuristic
        enabled (the default) -> the heuristic actually runs (not skipped)
        and the union winner is still a valid, finite T-vector."""
        import solvers.oracle as oracle_mod
        al = _make_hyb(shortlist=4, eps=0.01, oracle_head=True, oracle_max_n1=2,
                       oracle_heuristic_time_budget=0.05)
        self.assertTrue(al.oracle_use_heuristic)
        with mock.patch.object(oracle_mod, "t1_heuristic_argmin",
                               wraps=oracle_mod.t1_heuristic_argmin) as spy:
            torch.manual_seed(self.SEED)
            out = al(num_sample=self.K)
        spy.assert_called()
        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(np.isfinite(out).all())

    def test_enabled_end_to_end_reaches_oracle_T1(self):
        """oracle_head=True end-to-end: valid (1, 3) float64 output whose T1
        equals the exact optimum (neither arm can beat the oracle; the union
        winner is re-evaluated at float64)."""
        from solvers.oracle import t1_oracle_argmin
        al = _make_hyb(shortlist=4, eps=0.01, oracle_head=True, oracle_r=8,
                       oracle_shortlist=3)
        td0 = al.td[0]
        res = t1_oracle_argmin(
            td0["adj"].cpu().numpy().astype(np.float64),
            td0["service_times"].cpu().numpy().astype(np.float64),
            td0["clss"].cpu().numpy().astype(np.int64),
            td0["demand"].cpu().numpy().astype(np.float64),
            int(td0["num_vehicle"]), tiebreak=True)

        torch.manual_seed(self.SEED)
        out = al(num_sample=self.K)
        self.assertEqual(out.shape, (1, 3))
        self.assertEqual(out.dtype, np.float64)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isclose(out[0, 0], res["T1"], rtol=1e-5),
                        f"union T1 {out[0, 0]} != oracle {res['T1']}")


class TestUpgradeToU(unittest.TestCase):
    """`upgrade_to_u`: the zero-training post-processing pass that runs the
    unrestricted (variant-U) local-search kernels on the funnel's FINAL
    variant-P winner and keeps the result only if lex-better.

    Pins down the two contract points:
      1. default-off is byte-identical to before -- the upgrade code is never
         entered, so the flag-off output on a fixed seed equals the output of
         a solver constructed without the flag at all;
      2. the guard never lex-worsens -- on the same seed, the flag-on output
         is lex<= the flag-off output, and `last_upgrade_info` faithfully
         records whether the pass fired.
    """
    SEED = 123
    K = 16

    def test_default_off_and_code_not_reached(self):
        al = _make_hyb(shortlist=4, eps=0.01)
        self.assertFalse(al.upgrade_to_u)
        with mock.patch.object(al, "_upgrade_winner_to_u",
                               side_effect=AssertionError("must not run")) as m:
            torch.manual_seed(self.SEED)
            al(num_sample=self.K)
        m.assert_not_called()
        self.assertIsNone(al.last_upgrade_info)

    def test_default_off_output_identical(self):
        """Flag off (default) vs flag off explicitly: same seed -> byte-equal
        output (there is no code path difference at all)."""
        al0 = _make_hyb(shortlist=4, eps=0.01)
        torch.manual_seed(self.SEED)
        out0 = al0(num_sample=self.K)

        al1 = _make_hyb(shortlist=4, eps=0.01, upgrade_to_u=False)
        torch.manual_seed(self.SEED)
        out1 = al1(num_sample=self.K)
        self.assertTrue(np.array_equal(out0, out1))

    def test_guard_never_lex_worsens(self):
        """Same seed, flag on vs flag off: the upgrade pass consumes no RNG
        (deterministic local search after the final pick), so the two runs
        share the same pure-P winner -- the flag-on output must be lex<= it.
        `last_upgrade_info` must agree: pre_T == the flag-off output, post_T
        == the flag-on output, and `fired` iff they differ."""
        al_off = _make_hyb(shortlist=4, eps=0.01)
        torch.manual_seed(self.SEED)
        out_off = al_off(num_sample=self.K)[0]

        al_on = _make_hyb(shortlist=4, eps=0.01, upgrade_to_u=True)
        torch.manual_seed(self.SEED)
        out_on = al_on(num_sample=self.K)[0]

        self.assertLessEqual(tuple(out_on.tolist()), tuple(out_off.tolist()))
        info = al_on.last_upgrade_info
        self.assertIsNotNone(info)
        self.assertTrue(np.array_equal(info["pre_T"], out_off))
        self.assertTrue(np.array_equal(info["post_T"], out_on))
        self.assertEqual(info["fired"],
                         tuple(out_on.tolist()) < tuple(out_off.tolist()))

    def test_upgrade_kept_only_if_lex_better(self):
        """Force the upgrade candidate to be lex-WORSE than the winner: the
        guard must reject it (output = the pure flag-off winner, fired=False).
        The reference comes from a flag-off run: the real (lex-safe) upgrade
        pass can genuinely fire on this instance, so a flag-on run's output
        is not the pure winner the mocked-away pass should fall back to."""
        al_off = _make_hyb(shortlist=4, eps=0.01)
        torch.manual_seed(self.SEED)
        out_ref = al_off(num_sample=self.K)[0]

        al = _make_hyb(shortlist=4, eps=0.01, upgrade_to_u=True)
        worse = out_ref + 1.0
        with mock.patch.object(al, "_upgrade_winner_to_u",
                               return_value=(worse, None)):
            torch.manual_seed(self.SEED)
            out = al(num_sample=self.K)[0]
        self.assertTrue(np.array_equal(out, out_ref))
        self.assertFalse(al.last_upgrade_info["fired"])

    def test_upgrade_kept_when_lex_better(self):
        """Force a lex-better upgrade candidate: it must be returned and
        recorded as fired."""
        al = _make_hyb(shortlist=4, eps=0.01, upgrade_to_u=True)
        torch.manual_seed(self.SEED)
        out_ref = al(num_sample=self.K)[0]

        better = out_ref - 0.5
        with mock.patch.object(al, "_upgrade_winner_to_u",
                               return_value=(better, None)):
            torch.manual_seed(self.SEED)
            out = al(num_sample=self.K)[0]
        self.assertTrue(np.allclose(out, better))
        self.assertTrue(al.last_upgrade_info["fired"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
