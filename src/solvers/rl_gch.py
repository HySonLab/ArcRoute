"""Learning-free hybrid solver: HARMONY's pool funnel with the frozen policy
swapped out for the randomized Greedy Constructive Heuristic (GCH) as the
rollout source.

This is the "learning-free hybrid" ablation arm (paper Table `tab:ablation-
planned`, configuration A5). It answers a question distinct from the purely
learned baseline (Section on a purely learned baseline): does HARMONY's
inference-time pool funnel (class-1 candidates + guarded local search +
shortlist/final-selection) need a *learned* rollout source, or does *any*
diverse rollout source work just as well?

Design (fair isolation of the policy component):
  * `GCHHybHCARP` subclasses `RLHybHCARP` and REUSES its funnel verbatim --
    `_run_funnel_once` (shortlist -> guarded local search -> `final_band_pick`),
    `_local_search_one`, `_upgrade_winner_to_u`, `_oracle_forced_prefixes`
    (the class-1 candidate generator `t1_heuristic_candidates`), and the
    module-level `eps_lex_select`/`final_band_pick` are all inherited unchanged.
  * ONLY the rollout source is overridden: `_decode_pool` (free + forced
    rollouts) and `_greedy_anchor`. Where full HARMONY samples the frozen
    policy, this arm constructs solutions with GCH (`BaseHCARP.get_once`, the
    exact randomized constructor that seeds the ILS/EA baselines).
  * No policy checkpoint, no GPU, no training are used or loaded.

Rollout construction:
  * Free rollouts: `num_free` independent randomized GCH constructions, each a
    complete class-priority-respecting solution, converted to Joint M x N flat
    actions preserving GCH's own (vehicle, arc) assignment -- decoded downstream
    by the SAME `routes_from_joint_actions` full HARMONY uses for its policy
    rollouts.
  * Forced rollouts (HDCARP-P only): each class-1 candidate partition from
    `t1_heuristic_candidates` (identical to full HARMONY) fixes a class-1
    opening prefix; GCH then completes classes >= 2 on top of it. Under variant
    P's precedence, class-1 arcs are served first, so every completion of a
    fixed class-1 partition shares the SAME T1 -- exactly the invariant the
    funnel's forced-row lane assumes, so that lane runs unchanged. Each vehicle
    serves a single continuous trip under one capacity budget (no depot-reload
    separators) -- see `_gch_complete_forced` (reload-bug fix).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from numpy.random import permutation

from solvers.rl import RLHybHCARP
from solvers.evaluator import SolutionEvaluator
from solvers.meta import BaseHCARP


class GCHHybHCARP(RLHybHCARP):
    def __init__(self, variant, shortlist=25, eps=0.01, topk_t1=100,
                 oracle_head=True, oracle_k=3, oracle_r=64, oracle_shortlist=8,
                 oracle_union_eps2=0.02,
                 oracle_heuristic_time_budget=None,
                 oracle_heuristic_intra_cutoff=None, upgrade_to_u=False,
                 seed=6868):
        # NOTE: deliberately does NOT call super().__init__ -- that loads a
        # torch policy from a checkpoint. This arm has no policy. We replicate
        # only the funnel-parameter setup RLHybHCARP.__init__ performs.
        self.variant = variant
        self.shortlist = shortlist
        self.eps = eps
        self.topk_t1 = topk_t1
        self.oracle_head = oracle_head
        self.oracle_k = oracle_k
        self.oracle_r = oracle_r
        self.oracle_shortlist = oracle_shortlist
        self.oracle_union_eps2 = oracle_union_eps2
        from solvers.oracle import (HEURISTIC_INTRA_CUTOFF_DEFAULT,
                                    HEURISTIC_TIME_BUDGET_DEFAULT)
        self.oracle_heuristic_time_budget = (
            HEURISTIC_TIME_BUDGET_DEFAULT if oracle_heuristic_time_budget is None
            else oracle_heuristic_time_budget)
        self.oracle_heuristic_intra_cutoff = (
            HEURISTIC_INTRA_CUTOFF_DEFAULT if oracle_heuristic_intra_cutoff is None
            else oracle_heuristic_intra_cutoff)
        self.evaluator = SolutionEvaluator(variant=variant)
        self.upgrade_to_u = upgrade_to_u
        self.last_upgrade_info = None
        if upgrade_to_u:
            self._u_evaluator = SolutionEvaluator(variant='U')
        self.gch = BaseHCARP()
        self.gch.variant = variant
        self._rng = np.random.default_rng(seed)
        # for the module-shared TensorDict path
        from tensordict import TensorDict as _TD
        self._TensorDict = _TD

    # ------------------------------------------------------------------ data
    def _load_data(self, dms, P, veh, demands, clss, s, d, edge_indxs):
        """Load one instance into (a) a torch TensorDict the inherited funnel /
        evaluator read, and (b) a BaseHCARP the GCH constructor reads."""
        M_val = len(veh)
        self.td = self._TensorDict(
            {
                "clss":            torch.tensor(clss[None, :], dtype=torch.int64),
                "demand":          torch.tensor(demands[None, :], dtype=torch.float32),
                "demands":         torch.tensor(demands[None, :], dtype=torch.float32),
                "service_times":   torch.tensor(s[None, :], dtype=torch.float32),
                "traversal_times": torch.tensor(d[None, :], dtype=torch.float32),
                "adj":             torch.tensor(dms[None, :], dtype=torch.float32),
                "num_vehicle":     torch.tensor([[M_val]], dtype=torch.int64),
            },
            batch_size=[1],
        )
        self.gch._load_data(dms, P, veh, demands, clss, s, d, edge_indxs)

    # ----------------------------------------------------- rollout utilities
    def _arcorder_to_joint(self, a, N1):
        """GCH flat arc-order [v0 arcs, 0, v1 arcs, 0, ...] -> Joint M x N flat
        actions (v * N1 + arc), preserving GCH's own vehicle assignment."""
        joint, v = [], 0
        for x in np.asarray(a).ravel().tolist():
            x = int(x)
            if x == 0:
                v += 1
            else:
                joint.append(v * N1 + x)
        return np.array(joint, dtype=np.int64)

    def _gch_free_rollout(self, N1):
        """One randomized GCH construction as Joint M x N actions, or None."""
        a = self.gch.get_once(self.gch.M, self.gch.clss)
        if a is None:
            return None
        return self._arcorder_to_joint(a, N1)

    def _collect_free(self, num_free, N1):
        """Exactly `num_free` GCH free rollouts (retry on construction failure;
        pad by resampling if the instance is hard to construct feasibly)."""
        acts = []
        budget = max(num_free * 8, 32)
        while len(acts) < num_free and budget > 0:
            r = self._gch_free_rollout(N1)
            budget -= 1
            if r is not None and len(r):
                acts.append(r)
        if not acts:
            raise RuntimeError("GCH failed to construct any feasible rollout")
        while len(acts) < num_free:                     # pad to a fixed pool size
            acts.append(acts[len(acts) % max(len(acts), 1)])
        return acts[:num_free]

    def _joint_prefix_to_parts(self, forced, N1, M):
        """Inverse of oracle.forced_flat_actions: Joint prefix -> per-vehicle
        class-1 arc lists (in service order)."""
        parts = [[] for _ in range(M)]
        for a in np.asarray(forced).ravel().tolist():
            a = int(a)
            v, j = a // N1, a % N1
            if 0 <= v < M and j != 0:
                parts[v].append(j)
        return parts

    def _gch_complete_forced(self, parts, N1, M, greedy=False):
        """Complete a fixed class-1 partition with GCH-constructed classes >= 2.

        Class-1 arcs stay exactly where `t1_heuristic_candidates` put them (so
        T1 is identical across all completions of this candidate). Higher-class
        arcs are assigned to vehicles by a WHOLE-ROUTE-budget-respecting
        least-load rule: each higher-class arc goes to the least-loaded vehicle
        that still has room for it in ONE trip (greedy = the single least-loaded
        such vehicle; sampled = one of the three least-loaded among those with
        room). Each vehicle then serves a SINGLE continuous trip, class-ordered,
        with NO depot separators -- the reload-bug fix (see
        docs/root-cause-allow.md): the old bridge split each (vehicle, class)
        block through `_split_by_capacity` and emitted depot-return separators,
        i.e. deliberately constructed multi-trip reloads. If no vehicle has room
        for an arc the fleet is exhausted for a single-trip solution; the arc is
        placed on the least-loaded vehicle anyway so the whole-route budget
        check in `SolutionEvaluator` marks the completion infeasible and the
        funnel discards it (rather than silently reloading). Returns Joint M x N
        flat actions."""
        clss = self.gch.clss
        demands = self.gch.demands.astype(np.float64)
        classes = [p for p in self.gch.P]
        cap = 1.0

        # per-(vehicle, class) arc buckets; class-1 fixed from the candidate
        buckets = {v: {p: [] for p in classes} for v in range(M)}
        veh_load = np.zeros(M)
        for v in range(M):
            for arc in parts[v]:
                if 1 in buckets[v]:
                    buckets[v][1].append(int(arc))
                veh_load[v] += demands[int(arc)]

        for p in classes:
            if p <= 1:
                continue
            arcs = np.where(clss == p)[0]
            order = np.sort(arcs)[::-1] if greedy else permutation(arcs)
            for arc in order.tolist():
                arc = int(arc)
                d = demands[arc]
                room = np.where(veh_load + d <= cap + 1e-9)[0]
                if len(room) == 0:
                    v = int(np.argmin(veh_load))         # exhausted -> infeasible
                elif greedy:
                    v = int(room[np.argmin(veh_load[room])])
                else:
                    cand = room[np.argsort(veh_load[room])[:min(3, len(room))]]
                    v = int(self._rng.choice(cand))
                buckets[v][p].append(arc)
                veh_load[v] += d

        # One continuous single-trip route per vehicle, class-ordered. No depot
        # separators: a vehicle's arcs share ONE capacity budget (no reload).
        joint = []
        for v in range(M):
            for p in classes:
                for arc in buckets[v][p]:
                    joint.append(v * N1 + int(arc))
        return np.array(joint, dtype=np.int64)

    @staticmethod
    def _pad_stack(acts):
        S = max((len(a) for a in acts), default=1)
        S = max(S, 1)
        out = np.zeros((len(acts), S), dtype=np.int64)
        for i, a in enumerate(acts):
            out[i, :len(a)] = a
        return out

    # ---------------------------------------------- overridden rollout source
    def _decode_pool(self, num_free, forced):
        """GCH replacement for the policy decode pool: `num_free` free GCH
        rollouts, plus (when a class-1 prefix is forced) `1 + oracle_r` GCH
        completions of it. Returns (reward (B,P) float64, actions (B,S) int64,
        n_forced) -- same contract as RLHybHCARP._decode_pool."""
        td0 = self.td[0]
        M = max(int(td0["num_vehicle"]), 1)
        N1 = td0["service_times"].shape[0]
        self.evaluator._cache_td(td0)

        def score(joint):
            veh = self.evaluator.routes_from_joint_actions(joint, M, N1)
            return self.evaluator._completion_times(veh, td0)

        acts = self._collect_free(num_free, N1)
        n_forced = 0
        if forced is not None:
            parts = self._joint_prefix_to_parts(forced, N1, M)
            n_forced = 1 + self.oracle_r
            for i in range(n_forced):
                acts.append(self._gch_complete_forced(
                    parts, N1, M, greedy=(i == 0)))

        reward = np.stack([score(a) for a in acts]).astype(np.float64)
        actions = self._pad_stack(acts)
        return reward, actions, n_forced

    def _greedy_anchor(self):
        """GCH replacement for the policy greedy-decode anchor row: one extra
        GCH construction, scored through the same evaluator. Returns
        (g_reward (1,P) float64, g_actions (1,S) int64)."""
        td0 = self.td[0]
        M = max(int(td0["num_vehicle"]), 1)
        N1 = td0["service_times"].shape[0]
        self.evaluator._cache_td(td0)
        joint = None
        for _ in range(32):
            joint = self._gch_free_rollout(N1)
            if joint is not None and len(joint):
                break
        if joint is None or not len(joint):
            joint = np.array([1], dtype=np.int64)       # degenerate fallback
        veh = self.evaluator.routes_from_joint_actions(joint, M, N1)
        g_reward = self.evaluator._completion_times(veh, td0)[None, :].astype(np.float64)
        g_actions = joint[None, :].astype(np.int64)
        return g_reward, g_actions
