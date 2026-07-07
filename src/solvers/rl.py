"""RL (GRPO) solver for HDCARP.

Usage (single instance — route log):
    uv run python -m solvers.rl --ckpt outputs/checkpoints/last.ckpt \\
        --file data/ood/osm_cityA/40/38_17_358.npz \\
        --vehicles 2 --num_sample 100 --log outputs/rl_route_log.txt

Usage (batch benchmark):
    uv run python -m solvers.rl --ckpt outputs/checkpoints/last.ckpt \\
        --path data/5m --variant P --num_sample 1
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import contextlib
import io
from glob import glob
from time import time

import numpy as np
import torch

from utils.ops import batchify, import_instance, iter_instances, print_route_log, save_sol


def eps_lex_select(reward, eps=0.005, shortlist=1):
    """Select up to `shortlist` sample indices, ranked by (T2, T3) among the
    samples whose T1 is within `eps` (relative) of the best T1 in the group.

    Plain lexsort on continuous T-vectors degenerates to argmin-T1 only (exact
    ties never happen), so T2/T3 never act as a real tie-break. This widens
    the "tied" band to an eps-tolerance so near-T1-optimal samples can compete
    on T2/T3. Returns indices into `reward`'s first axis, best first.

    reward : (num_sample, P) array, LOWER is better.
    """
    t1 = reward[:, 0]
    thresh = t1.min() * (1.0 + eps) if t1.min() > 0 else t1.min() - abs(t1.min()) * eps - 1e-9
    keep = np.where(t1 <= thresh)[0]
    if len(keep) > 1:
        order = np.lexsort((reward[keep, 2], reward[keep, 1]))
        keep = keep[order]
    if len(keep) < shortlist:
        # pad with the next-best strict-lex candidates outside the eps band
        order_all = np.lexsort((reward[:, 2], reward[:, 1], reward[:, 0]))
        keep_set = set(keep.tolist())
        extra = [i for i in order_all if i not in keep_set]
        need = shortlist - len(keep)
        keep = np.concatenate([keep, np.array(extra[:need], dtype=keep.dtype)])
    return keep[:shortlist]


def final_band_pick(Ts, eps2, tie_tol=1e-6):
    """Single selection rule over a combined candidate pool (free rollouts
    plus, when applicable, oracle-forced rollouts) -- one rule covering what
    used to be a two-arm split (`t2_band_pick` + `oracle_union_pick`).

    T1 tie-band (relative `tie_tol`) -> among those, T2 eps2-band -> min T3.
    Degenerates to argmin-T1 when no two candidates' T1 are within `tie_tol`
    of each other (the common case among distinct free rollouts), and to a
    pure (T2, T3) choice when candidates share one forced T1 exactly (the
    oracle-forced-rows case).

    Ts : (k, 3) array of T-vectors, LOWER is better (non-finite T1 ignored).
    Returns a row index into `Ts`.
    """
    ok = np.where(np.isfinite(Ts[:, 0]))[0]
    c = Ts[ok]
    m1 = c[:, 0].min()
    b1 = np.where(c[:, 0] <= m1 + tie_tol * max(m1, 1e-9))[0]
    c2 = c[b1]
    b2 = b1[np.where(c2[:, 1] <= c2[:, 1].min() * (1.0 + eps2))[0]]
    return int(ok[b2[np.argmin(c[b2, 2])]])


class RLHCARP:
    def __init__(self, pw, variant, device='cuda'):
        from policy.policy import AttentionModelPolicy
        from env.env import CARPEnv
        from tensordict import TensorDict as _TD
        self._TensorDict = _TD
        ckpt = torch.load(pw, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        policy = AttentionModelPolicy(
            embed_dim=hp.get("embed_dim", 128),
            num_encoder_layers=hp.get("num_encoder_layers", 12),
            num_heads=hp.get("num_heads", 8),
        )
        sd = {k[len("policy."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("policy.")}
        policy.load_state_dict(sd, strict=True)
        self.device = device
        self.variant = variant
        self.policy = policy.to(device).eval()
        self.env = CARPEnv(variant=variant, reward_mode="vector")

    def _load_data(self, dms, P, veh, demands, clss, s, d, edge_indxs):
        M_val = len(veh)
        td = self._TensorDict(
            {
                "clss":            torch.tensor(clss[None, :], dtype=torch.int64),
                "demands":         torch.tensor(demands[None, :], dtype=torch.float32),
                "service_times":   torch.tensor(s[None, :], dtype=torch.float32),
                "traversal_times": torch.tensor(d[None, :], dtype=torch.float32),
                "adj":             torch.tensor(dms[None, :], dtype=torch.float32),
                "num_vehicle":     torch.tensor([[M_val]], dtype=torch.int64),
            },
            batch_size=[1],
        )
        self.td = self.env.reset(td).to(self.device)

    def __call__(self, num_sample=100, eps=0.005):
        td = batchify(self.td, num_sample)
        with torch.inference_mode():
            out = self.policy(td, env=self.env, decode_type='sampling')
        reward = out["reward"].cpu().numpy()          # (num_sample, P)
        idx = int(eps_lex_select(reward, eps=eps, shortlist=1)[0])
        return reward[idx:idx+1]                      # (1, P)


class RLHybHCARP(RLHCARP):
    """Single-pool funnel hybrid: one batched decode produces a combined pool
    of free-sampled rollouts plus (when applicable) oracle-forced-prefix
    rollouts, then ONE shortlist + local-search + selection funnel picks the
    final answer -- no separate arms, no bespoke union comparator.

    `_oracle_forced_prefix` computes a (provably-optimal-when-exact,
    empirically-near-optimal-when-heuristic) class-1 opening prefix per
    vehicle (variant P only); `_decode_pool` decodes `num_sample` free rows
    plus, when a prefix is available, `1 + oracle_r` prefix-forced rows
    (1 greedy + `oracle_r` sampled completions) in a SINGLE batched decode
    loop -- rows with an unconsumed forced action get it overridden each
    step, all other rows sample normally. Degenerate cases (variant U, no
    capacity-feasible class-1 partition, or an oracle/env mask mismatch mid-
    decode) collapse to 0 forced rows, i.e. exactly the plain pool funnel
    (and `_decode_pool` delegates straight to `AttentionModelPolicy.forward`
    in that case, so the default `oracle_head=False` path's rollout is
    unchanged from before the unification).

    A single deterministic GREEDY rollout is appended to the pool as an
    anchor row (always shortlisted, decoded after the sampling pool via
    argmax so no RNG is consumed): the pool is otherwise all sampled
    rollouts, and on OOD instances a policy with a flat sampled distribution
    can have every sample decode worse than its own greedy argmax (measured
    for variant U on osm_test_small -- see the comment in `__call__`), which
    used to make the funnel worse than a plain greedy decode. With the
    anchor the funnel is provably no worse than greedy as well.

    Local search (`utils.local_search`) then runs on a shortlist spanning the
    whole pool -- `eps_lex_select` over everything, `topk_t1` over the free
    rows (a T1-only hedge against an imperfect forced prefix), a guaranteed
    lane of the best forced rows by (T2, T3) when forcing engaged, plus the
    greedy anchor -- and only keeps a candidate's post-LS T-vector if it's
    lex-better than pre-LS, so the hybrid is provably no worse than plain
    `RLHCARP`. Under variant U, `_local_search_one` runs BOTH the
    unrestricted U arm and the class-restricted P arm and keeps the
    lex-better (every P-legal move is U-legal -- same free win as
    `BaseHCARP.ls_lex_guarded`). The final `final_band_pick` (T1 tie-band ->
    T2 eps-band -> min T3) picks the winner, re-evaluated through the
    float64 `SolutionEvaluator`.

    Unified (see `docs/grpo.md` §17/§18) from an earlier two-arm design
    (separate pool-arm and oracle-head-arm decodes, joined by a since-removed
    `oracle_union_pick`) after paired-seed A/B validation on 820 real
    instances (`outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt`) found no
    regression versus the two-arm production pipeline, in either the
    exact-DP regime (n1<=16, 520 instances) or the heuristic-fallback regime
    (n1 19-35, 300 instances): T1 never regressed (0/820), and T2/T3 scatter
    sat at or below the pipeline's own seed-noise floor, while running
    ~8-20% faster (one shared encoder pass instead of two).

    Historical validation this design carries forward (`data/osm_test_small`,
    2858 instances, same checkpoint, see `logs/status_bm.md`):
    - `topk_t1` (T1-only hedge lane): raising `num_sample` 100->400 and
      adding `topk_t1=100` moved "T1 not worse than ILS" 82.75%->95.31%, "T2
      not worse" 44.2%->53.1%, lex-win 69.3%->86.4%, Pareto-dominance
      26.2%->36.5%, at 0.144s/instance (5.6x faster than ILS).
    - `oracle_head` (forced-prefix rows, exact DP when n1<=`oracle_max_n1`):
      on top of the above, "T1 not worse than ILS" 95.31%->100%, T2
      52.9%->86.0%, T3 55.9%->63.5%, lex-win 86.4%->98.3%, Pareto
      36.5%->57.7%, at ~3.4x ILS speed.
    - `oracle_use_heuristic` (`t1_heuristic_argmin` fallback above
      `oracle_max_n1`, exact DP being `O(2^n1 * n1^2)`): essentially exact
      (<0.1% mean gap) through n1~20, degrading to ~1.7%/2.7%/5.6% mean gap
      at n1~25/31/35, well under 1s/instance throughout. The forced-prefix
      lane's guarantee degrades accordingly from "provably optimal" to
      "empirically within a few percent of best-found" at that scale, but
      the funnel still only ever keeps a forced candidate if it's lex-better
      than or tied with the rest of the pool -- it can only help or tie,
      never hurt (verified pre-unification: 0 T1-safety violations across
      12,780 union decisions built from real exact-vs-heuristic (T1) pairs).
    """

    def __init__(self, pw, variant, device='cuda', shortlist=10, eps=0.005, topk_t1=0,
                 oracle_head=False, oracle_r=64, oracle_shortlist=8,
                 oracle_union_eps2=0.02,
                 oracle_tiebreak=True, oracle_max_n1=None,
                 oracle_use_heuristic=True, oracle_heuristic_time_budget=None,
                 oracle_heuristic_intra_cutoff=None, upgrade_to_u=False):
        super().__init__(pw, variant, device=device)
        self.shortlist = shortlist
        self.eps = eps
        self.topk_t1 = topk_t1
        self.oracle_head = oracle_head
        self.oracle_r = oracle_r
        self.oracle_shortlist = oracle_shortlist
        self.oracle_union_eps2 = oracle_union_eps2
        self.oracle_tiebreak = oracle_tiebreak
        from solvers.oracle import (HEURISTIC_INTRA_CUTOFF_DEFAULT,
                                    HEURISTIC_TIME_BUDGET_DEFAULT, MAX_N1_DEFAULT)
        if oracle_max_n1 is None:
            oracle_max_n1 = MAX_N1_DEFAULT
        self.oracle_max_n1 = oracle_max_n1
        self.oracle_use_heuristic = oracle_use_heuristic
        if oracle_heuristic_time_budget is None:
            oracle_heuristic_time_budget = HEURISTIC_TIME_BUDGET_DEFAULT
        self.oracle_heuristic_time_budget = oracle_heuristic_time_budget
        if oracle_heuristic_intra_cutoff is None:
            oracle_heuristic_intra_cutoff = HEURISTIC_INTRA_CUTOFF_DEFAULT
        self.oracle_heuristic_intra_cutoff = oracle_heuristic_intra_cutoff
        from solvers.evaluator import SolutionEvaluator
        self.evaluator = SolutionEvaluator(variant=variant)
        # upgrade_to_u: zero-training variant-U answer from the variant-P
        # pipeline. After the normal funnel picks its final winner, ONE extra
        # unrestricted (variant-U) local-search pass runs on that winner --
        # any-class swap/relocate moves that P's own class-restricted search
        # was never allowed to make -- with MOVE-LEVEL lex-safety (each move
        # is accepted only if the full (T1,T2,T3) does not lex-worsen; see
        # `u_route_opt_lex_safe`), re-scored as a U solution and kept only
        # if lex-better (guard: can only help or tie, never hurt). Off by
        # default: the flag-off path is byte-identical to before (no extra
        # code runs, no RNG consumed). See `_upgrade_winner_to_u`.
        self.upgrade_to_u = upgrade_to_u
        self.last_upgrade_info = None
        if upgrade_to_u:
            # The upgraded routes may interleave classes, so they must be
            # scored as a U solution. (`_completion_times` itself is variant-
            # agnostic -- it scores whatever vehicle/trip structure it is
            # given -- but keep a dedicated U evaluator so the scoring intent
            # is explicit and survives any future variant-dependent scoring.)
            self._u_evaluator = SolutionEvaluator(variant='U')

    def _local_search_one(self, vehicles, td0, M, adj, clss, demands, classes):
        """Run intra/inter local search on a decoded vehicles structure.

        Variant P runs the class-restricted arm. Variant U runs BOTH the
        unrestricted U arm and the class-restricted P arm and keeps the
        lex-better result (U arm preferred on exact ties) -- mirroring
        `BaseHCARP.ls_lex_guarded` (solvers/meta.py): every P-legal move is
        also U-legal, so this is a free win. Without it, the unrestricted arm
        often trades T1 away for deadhead, the caller's lex guard rejects it
        outright, and a U candidate gets NO local-search benefit at all where
        a P candidate would have kept its class-safe T2/T3 gains.

        Returns (T_vec, new_vehicles), or (None, None) when there is nothing
        to optimize.
        """
        from utils.local_search import intra_route_opt, inter_route_opt

        trips, owner = [], []
        for m, veh_trips in enumerate(vehicles):
            for trip in veh_trips:
                trips.append(np.concatenate([[0], trip, [0]]).astype(np.int64))
                owner.append(m)
        if not trips:
            return None, None

        def _arm(variant):
            # `trips` is never mutated (intra/inter both copy), so each arm
            # starts from the same pre-LS routes.
            if variant == 'P':
                new_trips = []
                for route in trips:
                    for p in classes:
                        route = intra_route_opt(route, adj, clss, variant='P', p=p)
                    new_trips.append(route)
            else:
                new_trips = [intra_route_opt(r, adj, clss, variant='U')
                             for r in trips]
            new_trips = inter_route_opt(new_trips, adj, demands, clss,
                                        capacity=1.0, variant=variant)
            new_vehicles = [[] for _ in range(M)]
            for m, route in zip(owner, new_trips):
                trip = route[1:-1]
                if len(trip):
                    new_vehicles[m].append(trip)
            return self.evaluator._completion_times(new_vehicles, td0), new_vehicles

        best_T, best_vehicles = _arm(self.variant)
        if self.variant == 'U':
            p_T, p_vehicles = _arm('P')
            if tuple(p_T.tolist()) < tuple(best_T.tolist()):
                best_T, best_vehicles = p_T, p_vehicles
        return best_T, best_vehicles

    def _upgrade_winner_to_u(self, vehicles, td0, M, adj, clss, demands):
        """Move-level lex-safe unrestricted (variant-U) local-search pass over
        an already-final winner (see `upgrade_to_u` in `__init__`):
        `u_route_opt_lex_safe` with the variant-U evaluator supplying the
        per-vehicle objective. The caller still applies the whole-route lex
        guard as a final safety net; this only produces the candidate.

        Why not the plain deadhead-greedy U kernels (`intra_route_opt` +
        `inter_route_opt`, variant='U'): measured on data/test_bm/small.npz
        under the production config, that pass reduced total deadhead on
        100/100 instances (mean -11.94) yet the caller's guard rejected it
        100/100 times because it always bundled at least one T1-worsening
        move (T1 worse on 100/100, often 2-3x) with the harmless ones. The
        lex-safe search accepts/rejects each MOVE on the full (T1, T2, T3)
        tuple instead, so T1 is monotone non-increasing across the pass by
        construction and the T1-neutral T2/T3 headroom becomes reachable.

        Mechanism ceiling (by construction): this reorders/relocates arcs
        within (and between) the trips the P decode already produced -- it
        cannot discover a fundamentally different vehicle-arc assignment that
        only a genuinely U-aware decode would reach.

        Returns (T_vec, new_vehicles), or (None, None) when there is nothing
        to optimize.
        """
        from utils.local_search import u_route_opt_lex_safe

        trips, owner = [], []
        for m, veh_trips in enumerate(vehicles):
            for trip in veh_trips:
                trips.append(np.concatenate([[0], trip, [0]]).astype(np.int64))
                owner.append(m)
        if not trips:
            return None, None

        self._u_evaluator._cache_td(td0)

        def veh_score(veh_trips):
            # ONE vehicle's per-class max completion times; the elementwise
            # max over vehicles equals scoring the whole solution at once.
            return self._u_evaluator._completion_times([veh_trips], td0)

        new_trips = u_route_opt_lex_safe(trips, owner, M, adj, demands,
                                         veh_score, capacity=1.0)
        new_vehicles = [[] for _ in range(M)]
        for m, route in zip(owner, new_trips):
            trip = route[1:-1]
            if len(trip):
                new_vehicles[m].append(trip)
        return self._u_evaluator._completion_times(new_vehicles, td0), new_vehicles

    def _oracle_forced_prefix(self, td0, N1, adj, clss, demands, M):
        """Compute the oracle's forced class-1 opening-prefix flat actions,
        or None when not applicable (variant U, `oracle_head=False`, n1 too
        large with the heuristic disabled, or no capacity-feasible partition
        exists). Exact DP (`t1_oracle_argmin`) when n1 <= `oracle_max_n1`;
        otherwise `t1_heuristic_argmin` if `oracle_use_heuristic` (see the
        class docstring for the measured quality/speed tradeoff of that
        fallback).
        """
        from solvers.oracle import (forced_flat_actions, t1_heuristic_argmin,
                                    t1_oracle_argmin)
        if not self.oracle_head or self.variant != 'P':
            # The oracle's premise (class-1 arcs form a forced opening prefix)
            # only holds under variant P's precedence; decline otherwise.
            return None
        svc = td0["service_times"].cpu().numpy().astype(np.float64)
        n1 = int((clss == 1).sum())
        if n1 <= self.oracle_max_n1:
            res = t1_oracle_argmin(adj, svc, clss, demands, M,
                                   tiebreak=self.oracle_tiebreak,
                                   max_n1=self.oracle_max_n1)
        elif self.oracle_use_heuristic:
            res = t1_heuristic_argmin(
                adj, svc, clss, demands, M,
                time_budget=self.oracle_heuristic_time_budget,
                intra_cutoff=self.oracle_heuristic_intra_cutoff)
        else:
            return None
        if not np.isfinite(res["T1"]) or res["parts"] is None:
            return None                       # not applicable / infeasible
        return forced_flat_actions(res["parts"], N1)

    @torch.inference_mode()
    def _decode_pool(self, num_free, forced):
        """One batched decode producing the combined candidate pool:
        `num_free` free-sampled rollouts plus, when `forced` is not None,
        `1 + oracle_r` oracle-prefix-forced rows (1 greedy + `oracle_r`
        sampled completions) -- all decoded in a SINGLE loop. At each step,
        rows with an unconsumed forced action get it overridden onto the
        batch's action tensor before `env.step`; every other row (free rows,
        and forced rows past their prefix) samples normally.

        When `forced` is None (oracle not applicable), delegates straight to
        `AttentionModelPolicy.forward` -- this keeps the `oracle_head=False`
        default path's rollout byte-identical to before the pool/oracle-head
        unification (same RNG consumption), not just qualitatively similar.

        If the env's `action_mask` ever rejects the forced action (an
        oracle/env mismatch -- never observed on the benchmark), forcing is
        aborted for the rest of the decode and `n_forced` is reported as 0:
        those rows simply continue as ordinary samples and compete in the
        pool like any other rollout, rather than aborting the whole call.

        Returns (reward (B, P) float64 np, actions (B, S) int64 np, n_forced).
        """
        if forced is None:
            td = batchify(self.td, num_free)
            with torch.inference_mode():
                out = self.policy(td, env=self.env, decode_type='sampling')
            reward = out["reward"].cpu().numpy().astype(np.float64)
            actions = out["actions"].cpu().numpy().astype(np.int64)
            return reward, actions, 0

        n_forced = 1 + self.oracle_r
        B = num_free + n_forced
        td = batchify(self.td, B)
        hidden, _ = self.policy.encoder(td)
        td, env, cache = self.policy.decoder.pre_decoder_hook(td, self.env, hidden)
        L = len(forced)
        g = num_free                          # greedy-completion forced row
        forced_ok = True
        acts, t = [], 0
        while not td["done"].all():
            logits, mask = self.policy.decoder(td, cache)
            lf = logits.reshape(B, -1).masked_fill(~mask.reshape(B, -1), float('-inf'))
            sel = torch.multinomial(torch.softmax(lf, -1), 1).squeeze(1)
            if forced_ok:
                if t < L:
                    a = int(forced[t])
                    if not bool(mask.reshape(B, -1)[g, a]):
                        forced_ok = False      # oracle/env mismatch: demote to free
                    else:
                        sel[g:] = a
                else:
                    sel[g] = lf[g].argmax(-1)  # greedy completion row
            td.set("action", sel)
            td = env.step(td)
            acts.append(sel)
            t += 1
            if t > 10000:
                raise RuntimeError("decode did not terminate")
        reward = env.get_reward(td).cpu().numpy().astype(np.float64)
        actions = torch.stack(acts, 1).cpu().numpy().astype(np.int64)
        return reward, actions, (n_forced if forced_ok else 0)

    def __call__(self, num_sample=100):
        td0 = self.td[0]
        M = int(td0["num_vehicle"])
        N1 = td0["service_times"].shape[0]
        adj = td0["adj"].cpu().numpy().astype(np.float64)
        clss = td0["clss"].cpu().numpy().astype(np.int64)
        demands = td0["demand"].cpu().numpy().astype(np.float64)
        classes = [c for c in np.unique(clss) if c != 0]

        forced = self._oracle_forced_prefix(td0, N1, adj, clss, demands, M)
        reward, actions, n_forced = self._decode_pool(num_sample, forced)
        B = reward.shape[0]

        # ---- greedy anchor row ---------------------------------------------
        # One deterministic greedy rollout appended to the pool (and always
        # shortlisted), so the funnel is provably no worse than a plain greedy
        # decode. Rationale: the pool is otherwise all SAMPLED rollouts, and a
        # policy whose sampled distribution is flat on OOD instances can have
        # every one of its samples decode worse than its own greedy argmax --
        # measured for the variant-U checkpoint on osm_test_small: greedy
        # T1=5.97 vs min-of-400-samples T1=7.00, making the un-anchored funnel
        # lex-worse than greedy on 37/48 diagnostic instances. This mirrors
        # what the oracle head already gives variant P (its forced lane
        # carries a greedy-completion row). Decoded AFTER the sampling pool
        # and via argmax (no RNG consumed), so the sampled rows' RNG stream --
        # and `_decode_pool`'s byte-identical-to-plain-policy guarantee -- are
        # untouched.
        with torch.inference_mode():
            g_out = self.policy(batchify(self.td, 1), env=self.env,
                                decode_type='greedy')
        g_reward = g_out["reward"].cpu().numpy().astype(np.float64)
        g_actions = g_out["actions"].cpu().numpy().astype(np.int64)
        S = max(actions.shape[1], g_actions.shape[1])

        def _pad(a):
            # flat action 0 = (vehicle 0, depot): a no-op trip separator for
            # `routes_from_joint_actions`, safe as right-padding.
            return np.pad(a, ((0, 0), (0, S - a.shape[1])))

        reward = np.concatenate([reward, g_reward], axis=0)
        actions = np.concatenate([_pad(actions), _pad(g_actions)], axis=0)

        # ---- ONE shortlist over the combined pool --------------------------
        keep = set(int(i) for i in
                   eps_lex_select(reward, eps=self.eps, shortlist=self.shortlist))
        keep.add(B)                            # greedy anchor: always a candidate
        if self.topk_t1 > 0:
            # T1-only hedge lane, free rows only: forced rows all share one
            # T1, so ranking them by T1 is meaningless and would just crowd
            # out the hedge against an imperfect forced prefix.
            top = np.argsort(reward[:num_sample, 0])[:self.topk_t1]
            keep |= set(int(x) for x in top)
        if n_forced:
            # Guaranteed representation of the oracle candidates even when
            # the forced T1 doesn't anchor the eps band (the heuristic-oracle
            # case): the greedy row + the best `oracle_shortlist` forced rows
            # by (T2, T3) -- all forced rows share one T1, so this is a pure
            # (T2, T3) ranking (same rationale as the old, now-removed
            # `shortlist_completions`).
            fr = np.arange(num_sample, B)
            order = fr[np.lexsort((reward[fr, 2], reward[fr, 1]))]
            keep |= {num_sample} | set(int(i) for i in order[:self.oracle_shortlist])
        keep = sorted(keep)

        # SolutionEvaluator only lazily caches (adj/svc/clss) on first use, and
        # we call `_completion_times` directly below (bypassing `__call__`,
        # which is what normally refreshes the cache) -- must refresh
        # explicitly per instance or later, differently-sized instances read
        # stale arrays.
        self.evaluator._cache_td(td0)

        # ---- ONE local-search funnel + ONE final comparator ----------------
        # Every shortlisted candidate -- pre-LS included -- is scored through
        # the same float64 `SolutionEvaluator._completion_times` local search
        # uses and the final answer is reported in. Scoring pre-LS candidates
        # with the decode-time fp32 env reward instead (as before) made the
        # lex guard and `final_band_pick` compare apples to oranges: the env
        # scores a vehicle's trips in decode order while the evaluator
        # class-sorts them, so a candidate could win selection on its env
        # reward yet re-evaluate lex-worse than a candidate it beat (measured
        # under variant U: 8/48 diagnostic instances still came out worse
        # than the greedy anchor that was sitting in the same pool).
        cands = []                                  # (T_vec, vehicles)
        for idx in keep:
            pre_vehicles = self.evaluator.routes_from_joint_actions(actions[idx], M, N1)
            pre_T = self.evaluator._completion_times(pre_vehicles, td0)
            ls_T, ls_vehicles = self._local_search_one(
                pre_vehicles, td0, M, adj, clss, demands, classes)
            # local search optimizes total deadhead (a proxy for T_k), so only
            # keep it if it's actually lex-better than the pre-LS rollout.
            if ls_T is not None and tuple(ls_T.tolist()) < tuple(pre_T.tolist()):
                cands.append((np.asarray(ls_T, dtype=np.float64), ls_vehicles))
            else:
                cands.append((pre_T, pre_vehicles))
        Ts = np.stack([c[0] for c in cands])
        j = final_band_pick(Ts, self.oracle_union_eps2)

        # cands[j][0] is already the winner's float64 evaluator score (the
        # same basis every candidate was compared on), so the T1 optimality
        # guarantee when the winner came from the forced lane holds at full
        # precision, not just the decode-time fp32 reward's.
        best_T = np.asarray(cands[j][0], dtype=np.float64)

        # ---- optional zero-training U upgrade (see __init__) ---------------
        # Runs AFTER the funnel's final pick, on the winner only. Deterministic
        # (no RNG), guarded (kept only if lex-better), and skipped entirely
        # when the flag is off -- the default path above is untouched.
        if self.upgrade_to_u:
            up_T, _ = self._upgrade_winner_to_u(
                cands[j][1], td0, M, adj, clss, demands)
            fired = (up_T is not None
                     and tuple(up_T.tolist()) < tuple(best_T.tolist()))
            self.last_upgrade_info = {
                "fired": fired,
                "pre_T": best_T.copy(),
                "post_T": (np.asarray(up_T, dtype=np.float64)
                           if fired else best_T.copy()),
            }
            if fired:
                best_T = np.asarray(up_T, dtype=np.float64)

        return best_T[None, :]                                         # (1, P)


def parse_args():
    parser = argparse.ArgumentParser(description="RLHCARP")
    parser.add_argument('--seed', type=int, default=6868)
    parser.add_argument('--variant', default='P', choices=['P', 'U'])
    parser.add_argument('--num_sample', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.005,
                        help='relative T1 tolerance for eps-lex selection (default 0.5%%)')
    # single-instance log mode
    parser.add_argument('--ckpt', default=None, help='checkpoint (.ckpt)')
    parser.add_argument('--file', default=None, help='.npz instance; enables log mode')
    parser.add_argument('--vehicles', type=int, default=None, help='override fleet size')
    parser.add_argument('--log', default='outputs/rl_route_log.txt')
    parser.add_argument('--solution', default=None)
    # batch benchmark mode
    parser.add_argument('--path', default='data/5m', help='directory of .npz instances')
    parser.add_argument('--M', type=int, default=None)
    parser.add_argument('--hybrid', action='store_true',
                        help='batch mode: single-pool funnel (eps-lex shortlist + local search)')
    parser.add_argument('--shortlist', type=int, default=10,
                        help='hybrid mode: #candidates that get local search')
    parser.add_argument('--topk_t1', type=int, default=0,
                        help='hybrid mode: extra #candidates picked by raw T1 rank '
                             '(on top of the eps-lex shortlist), local-searched too. '
                             '0 = disabled (default, no behavior change)')
    parser.add_argument('--oracle_head', action='store_true',
                        help='hybrid mode: add oracle-forced-prefix rows (variant P) '
                             'into the same decode pool as the free rollouts '
                             '(default: off, no behavior change). Sub-tunables '
                             '(sample count, shortlist sizes, eps-bands, tiebreak) are '
                             'not exposed on the CLI -- historical sweeps never varied '
                             'them from RLHybHCARP\'s defaults; construct RLHybHCARP '
                             'directly if you need to.')
    parser.add_argument('--upgrade_to_u', action='store_true',
                        help='hybrid mode: after the funnel picks its final winner, '
                             'run ONE unrestricted (variant-U) local-search pass on '
                             'it and keep the result only if lex-better -- a zero-'
                             'training variant-U answer from a variant-P decode '
                             '(default: off, no behavior change)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.ckpt

    if args.file:
        # ── single-instance log mode ───────────────────────────────────────
        from env.env import CARPEnv
        from policy.policy import AttentionModelPolicy
        from solvers.meta import InsertCheapestHCARP
        from solvers.evaluator import SolutionEvaluator
        from tensordict import TensorDict

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        policy = AttentionModelPolicy(
            embed_dim=hp.get("embed_dim", 128),
            num_encoder_layers=hp.get("num_encoder_layers", 12),
            num_heads=hp.get("num_heads", 8),
        )
        sd = {k[len("policy."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("policy.")}
        policy.load_state_dict(sd, strict=True)
        policy = policy.to(device).eval()

        es = np.load(args.file)
        dms, P, M, demands, clss, s, d, _ = import_instance(args.file, M=args.vehicles)
        M = args.vehicles or M

        env = CARPEnv(
            num_loc=hp.get("num_loc", 40),
            num_arc=hp.get("num_arc", 80),
            variant=args.variant,
            reward_mode="vector",
        )
        td = TensorDict(
            {
                "clss": torch.tensor(clss[None, :], dtype=torch.int64),
                "demands": torch.tensor(demands[None, :], dtype=torch.float32),
                "service_times": torch.tensor(s[None, :], dtype=torch.float32),
                "traversal_times": torch.tensor(d[None, :], dtype=torch.float32),
                "adj": torch.tensor(dms[None, :], dtype=torch.float32),
                "num_vehicle": torch.tensor([[M]], dtype=torch.int64),
            },
            batch_size=[1],
        )
        td = env.reset(td).to(device)

        t0 = time()
        td_k = batchify(td, args.num_sample)
        with torch.inference_mode():
            out = policy(td_k, env=env, decode_type='sampling')
        Ts_all = out["reward"].cpu().numpy()
        idx = int(eps_lex_select(Ts_all, eps=args.eps, shortlist=1)[0])
        best_T = Ts_all[idx]
        best_actions = out["actions"][idx].cpu().numpy().astype(np.int32)
        elapsed = time() - t0

        evaluator = SolutionEvaluator(variant=args.variant)
        vehicles, best_T = evaluator(best_actions, td[0], M=M)
        best_routes = []
        for trips in vehicles:
            route = [0]
            for trip in trips:
                route.extend(int(a) for a in trip)
                route.append(0)
            if len(route) > 2:
                best_routes.append(np.array(route, dtype=np.int32))

        ils = InsertCheapestHCARP()
        ils.import_instance(args.file, M=M)

        req = es["req"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(f"Instance  : {args.file}")
            print(f"Checkpoint: {ckpt_path}")
            print(f"P={P} classes | M={M} vehicles | "
                  f"{len(req)} required arcs | {args.num_sample} samples | "
                  f"solved in {elapsed:.3f}s\n")
            print_route_log(best_routes, req, ils.demands, ils.vars["adj"], best_T)
        text = buf.getvalue()
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as fh:
            fh.write(text)
        print(text)
        print(f"Saved -> {args.log}")
        sol_path = args.solution or (os.path.splitext(args.log)[0] + ".sol")
        save_sol(sol_path, best_routes, best_T,
                 instance=args.file, checkpoint=ckpt_path,
                 variant=args.variant, vehicles=M)
        print(f"Saved -> {sol_path}")
    else:
        # ── batch benchmark mode ───────────────────────────────────────────
        # --path may be: a single .npz file, a flat dir of .npz files (e.g.
        # data/test_bm), or a nested city/bucket.npz dir (osm_test*).
        if args.path.endswith('.npz') and os.path.isfile(args.path):
            files = [args.path]
        else:
            files = sorted(glob(args.path + '/*/*.npz')) or sorted(glob(args.path + '/*.npz'))
        if args.hybrid:
            al = RLHybHCARP(ckpt_path, args.variant, shortlist=args.shortlist,
                            eps=args.eps, topk_t1=args.topk_t1,
                            oracle_head=args.oracle_head,
                            upgrade_to_u=args.upgrade_to_u)
        else:
            al = RLHCARP(ckpt_path, args.variant)
        for f in files:
            for i, data in enumerate(iter_instances(f, M=args.M)):
                al._load_data(*data)
                label = f if i == 0 else f"{f}[{i}]"
                t1 = time()
                if args.hybrid:
                    result = al(num_sample=args.num_sample)
                else:
                    result = al(num_sample=args.num_sample, eps=args.eps)
                print(label, ':::', result, ':::', time() - t1)