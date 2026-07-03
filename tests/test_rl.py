"""Tests for the RL hybrid solver (`solvers.rl`) and the exact T1 oracle
head (`solvers.oracle`, P2).

Covers:
  1. TestT1OracleArgmin      — oracle T1 matches brute force; `parts` are
     internally consistent (coverage, capacity, completion formula);
     tiebreak preserves T1 and never worsens the secondary; guards
     (max_n1, capacity-infeasible, no class-1 arcs) return "not applicable".
  2. TestSelectionHelpers    — shortlist_completions / t2_band_pick /
     oracle_union_pick unit behavior.
  3. TestForcedDecode        — forcing the oracle's `parts` through the real
     env/policy machinery reproduces the oracle T1 (fp32 tolerance).
  4. TestOracleHeadRegression — oracle_head=False (default) is byte-identical
     to the pre-P2 RLHybHCARP pipeline on a fixed seed (backward-compat
     guarantee); the guard fallback is byte-identical too; oracle_head=True
     returns a T1-optimal answer end-to-end.

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


def _p1_reference_call(al, num_sample):
    """Frozen copy of RLHybHCARP.__call__ as of P1 (pre-oracle-head).

    The regression test compares this byte-for-byte against the production
    __call__ with oracle_head=False under an identical RNG seed — any change
    to RNG consumption or selection logic on the default path breaks it."""
    from utils.ops import batchify
    from solvers.rl import eps_lex_select

    td = batchify(al.td, num_sample)
    with torch.inference_mode():
        out = al.policy(td, env=al.env, decode_type="sampling")
    reward = out["reward"].cpu().numpy()
    actions = out["actions"].cpu().numpy().astype(np.int64)

    keep = eps_lex_select(reward, eps=al.eps, shortlist=al.shortlist)
    if al.topk_t1 > 0:
        top = np.argsort(reward[:, 0])[:al.topk_t1]
        keep = np.array(sorted(set(keep.tolist()) | set(int(x) for x in top)),
                        dtype=keep.dtype)

    td0 = al.td[0]
    M = int(td0["num_vehicle"])
    N1 = td0["service_times"].shape[0]
    adj = td0["adj"].cpu().numpy().astype(np.float64)
    clss = td0["clss"].cpu().numpy().astype(np.int64)
    demands = td0["demand"].cpu().numpy().astype(np.float64)
    classes = [c for c in np.unique(clss) if c != 0]
    al.evaluator._cache_td(td0)

    best_T, best_key = None, None
    for idx in keep:
        pre_T = reward[idx]
        pre_vehicles = al.evaluator.routes_from_joint_actions(actions[idx], M, N1)
        ls_T, _ = al._local_search_one(pre_vehicles, td0, M, adj, clss,
                                       demands, classes)
        if ls_T is not None and tuple(ls_T.tolist()) < tuple(pre_T.tolist()):
            cand_T = ls_T
        else:
            cand_T = pre_T
        key = tuple(cand_T.tolist())
        if best_key is None or key < best_key:
            best_key, best_T = key, cand_T
    return np.asarray(best_T, dtype=np.float64)[None, :]


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


# ----------------------------------------------------------------------- 2
class TestSelectionHelpers(unittest.TestCase):
    def test_shortlist_completions_band_and_pad(self):
        from solvers.rl import shortlist_completions
        reward = np.array([[1.0, 0.500, 0.9],
                           [1.0, 0.501, 0.5],
                           [1.0, 0.600, 0.1]])
        # eps-band on T2 = {0, 1}, sorted by T3 -> [1, 0]
        self.assertEqual(shortlist_completions(reward, 2, eps=0.01).tolist(), [1, 0])
        # band smaller than k -> padded with strict-(T2,T3) rest
        self.assertEqual(shortlist_completions(reward, 3, eps=0.01).tolist(), [1, 0, 2])

    def test_t2_band_pick(self):
        from solvers.rl import t2_band_pick
        Ts = np.array([[1.0, 0.500, 0.9],
                       [1.0, 0.505, 0.2],
                       [1.0, 0.600, 0.1],
                       [np.inf, np.inf, np.inf]])          # padding row ignored
        self.assertEqual(t2_band_pick(Ts, 0.02), 1)        # band {0,1}, min T3
        self.assertEqual(t2_band_pick(Ts, 0.0), 0)         # zero band -> argmin T2

    def test_oracle_union_pick(self):
        from solvers.rl import oracle_union_pick
        a = np.array([1.0, 2.0, 3.0])
        # head T1 strictly better / strictly worse
        self.assertTrue(oracle_union_pick(a, np.array([0.9, 9.0, 9.0]), 0.02))
        self.assertFalse(oracle_union_pick(a, np.array([1.1, 0.0, 0.0]), 0.02))
        # T1 tie, T2 within band -> T3 decides
        self.assertTrue(oracle_union_pick(a, np.array([1.0, 2.03, 2.0]), 0.02))
        self.assertFalse(oracle_union_pick(a, np.array([1.0, 2.03, 4.0]), 0.02))
        # T1 tie, T2 outside band -> T2 decides
        self.assertFalse(oracle_union_pick(a, np.array([1.0, 2.5, 0.0]), 0.02))
        self.assertTrue(oracle_union_pick(a, np.array([1.0, 1.5, 9.0]), 0.02))


# ----------------------------------------------------------------------- 3
class TestForcedDecode(unittest.TestCase):
    def test_forced_prefix_reproduces_oracle_T1(self):
        """Forcing the oracle's `parts` through the real env/policy decode
        yields decode-time T1 == oracle T1 (fp32 arithmetic tolerance), for
        both the greedy and every sampled completion."""
        from solvers.oracle import forced_flat_actions, t1_oracle_argmin
        torch.manual_seed(11)
        al = _make_hyb(shortlist=4, eps=0.01)
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

        g_rew, g_act = al._decode_forced(forced, 1, greedy=True)
        s_rew, s_act = al._decode_forced(forced, 8, greedy=False)
        for rew in (g_rew, s_rew):
            self.assertTrue(np.allclose(rew[:, 0], res["T1"], rtol=1e-5),
                            f"decode T1 {rew[:, 0]} != oracle {res['T1']}")
        # forced opening is verbatim in the action sequences
        for act in (g_act, s_act):
            self.assertEqual(act[:, :len(forced)].tolist(),
                             [forced] * act.shape[0])


# ----------------------------------------------------------------------- 4
class TestOracleHeadRegression(unittest.TestCase):
    SEED = 123
    K = 16

    def test_default_off(self):
        al = _make_hyb()
        self.assertFalse(al.oracle_head)

    def test_oracle_code_not_reached_when_disabled(self):
        al = _make_hyb(shortlist=4, eps=0.01)
        with mock.patch.object(al, "_oracle_head_arm",
                               side_effect=AssertionError("must not run")) as m:
            torch.manual_seed(self.SEED)
            al(num_sample=self.K)
        m.assert_not_called()

    def test_disabled_is_byte_identical_to_p1(self):
        """THE backward-compat guarantee: with oracle_head=False (default),
        __call__ is byte-identical to the frozen P1 pipeline on a fixed seed."""
        al = _make_hyb(shortlist=4, eps=0.01, topk_t1=3)
        torch.manual_seed(self.SEED)
        ref = _p1_reference_call(al, self.K)
        torch.manual_seed(self.SEED)
        got = al(num_sample=self.K)
        self.assertTrue(np.array_equal(ref, got), f"{ref} != {got}")

    def test_guard_fallback_is_byte_identical(self):
        """oracle_head=True but oracle not applicable (max_n1 exceeded) ->
        pool answer returned unchanged, byte-identical to the default path
        (the guard trips before any extra RNG is consumed)."""
        al = _make_hyb(shortlist=4, eps=0.01, oracle_head=True, oracle_max_n1=2)
        torch.manual_seed(self.SEED)
        ref = _p1_reference_call(al, self.K)
        torch.manual_seed(self.SEED)
        got = al(num_sample=self.K)
        self.assertTrue(np.array_equal(ref, got), f"{ref} != {got}")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
