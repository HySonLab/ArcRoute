"""RL (GRPO) solver for HDCARP.

Usage (single instance — route log):
    uv run python -m solvers.rl --ckpt outputs/checkpoints/last.ckpt \\
        --file data/ood/osm_cityA/40/38_17_358.npz \\
        --vehicles 2 --num_sample 100 --log outputs/rl_route_log.txt

Usage (batch benchmark):
    uv run python -m solvers.rl --cpkt outputs/checkpoints/last.ckpt \\
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

from utils.ops import batchify, gen_tours, import_instance, iter_instances, print_route_log, save_sol


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


def shortlist_completions(reward, k, eps=0.01):
    """Shortlist up to `k` forced-prefix completions for local search.

    All completions decoded from one forced class-1 prefix share T1 exactly
    (the prefix determines it), so strict-lex ranking degenerates to
    argmin-T2 -- the same degeneracy `eps_lex_select` fixes for T1. Rank by an
    eps-band on T2 sorted by T3 instead, padded with the strict-(T2,T3)-lex
    rest when the band is smaller than `k`.

    reward : (R, P) array of completion T-vectors, LOWER is better.
    Returns indices into `reward`'s first axis, best first.
    """
    t2 = reward[:, 1]
    band = np.where(t2 <= t2.min() * (1.0 + eps))[0]
    band = band[np.argsort(reward[band, 2])]
    if len(band) < k:
        rest = np.lexsort((reward[:, 2], reward[:, 1]))
        seen = set(band.tolist())
        band = np.concatenate(
            [band, [i for i in rest if i not in seen]]).astype(np.int64)
    return band[:k]


def t2_band_pick(Ts, eps2):
    """Pick one candidate among finished oracle-head candidates.

    All rows share T1, so this is a pure (T2, T3) choice: keep candidates with
    T2 <= T2min * (1 + eps2), take min T3 among them. Returns a row index.

    Ts : (k, 3) array of T-vectors (rows with non-finite T1 are ignored).
    """
    ok = np.where(np.isfinite(Ts[:, 0]))[0]
    c = Ts[ok]
    band = np.where(c[:, 1] <= c[:, 1].min() * (1.0 + eps2))[0]
    return int(ok[band[np.argmin(c[band, 2])]])


def oracle_union_pick(a, b, eps2, tie_tol=1e-6):
    """Union rule between the pool arm's final pick and the oracle-head arm's.

    a : P1 (pool-based funnel) T-vector;  b : oracle-head T-vector.
    Returns True iff the head candidate `b` should be taken:
      - head T1 strictly better beyond a relative tie tolerance -> head;
      - P1 T1 strictly better -> P1;
      - T1 tied -> eps2-band on T2, min T3 (same band rule as `t2_band_pick`).
    """
    if b[0] < a[0] - tie_tol * max(a[0], 1e-9):
        return True
    if b[0] > a[0] + tie_tol * max(a[0], 1e-9):
        return False
    lo = min(a[1], b[1])
    if abs(a[1] - b[1]) <= eps2 * max(lo, 1e-9):
        return b[2] < a[2]
    return b[1] < a[1]


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

    def import_instance(self, f, M=None):
        self._load_data(*next(iter_instances(f, M=M)))

    def __call__(self, num_sample=100, eps=0.005):
        td = batchify(self.td, num_sample)
        with torch.inference_mode():
            out = self.policy(td, env=self.env, decode_type='sampling')
        reward = out["reward"].cpu().numpy()          # (num_sample, P)
        idx = int(eps_lex_select(reward, eps=eps, shortlist=1)[0])
        return reward[idx:idx+1]                      # (1, P)


def vehicles_to_action(vehicles):
    """Flatten a per-vehicle trips structure into the metaheuristics' flat
    arc-order action format.

    vehicles : list (len M) of lists-of-trips, each trip a 1D array of arc
               indices (what `SolutionEvaluator.routes_from_joint_actions`
               returns).
    Returns  : 1D np.int32 array of arc indices with a single 0 separating
               consecutive trips and no leading/trailing separator — the exact
               convention `get_once`/`deserialize_tours` (meta.py, utils/ops.py)
               produce and `gen_tours`/`ls`/`calc_obj` consume. Trips of the
               same vehicle are separated exactly like trips of different
               vehicles (the 0-separated format carries no vehicle identity;
               for variant P the evaluator re-derives the vehicle assignment
               from the arc order alone via `_build_P`).
    """
    flat = []
    for veh_trips in vehicles:
        for trip in veh_trips:
            arcs = [int(a) for a in np.asarray(trip).ravel().tolist() if a != 0]
            if arcs:
                flat.extend(arcs)
                flat.append(0)
    if flat and flat[-1] == 0:
        flat.pop()
    return np.array(flat, dtype=np.int32)


class RLHybHCARP(RLHCARP):
    """Two-tier funnel hybrid: sample K rollouts, shortlist the ones with
    near-best T1 (eps-lex), then run exact intra/inter-route local search
    (`utils.local_search`) on just the shortlist and keep the lex-best result.

    Local search minimizes total deadheading cost, a proxy for the true
    max-completion (T_k) objective, so a candidate's post-LS T-vector is only
    kept if it's lex-better than its pre-LS T-vector -- this makes the hybrid
    provably no worse than plain `RLHCARP` with the same eps/num_sample.

    `topk_t1` (default 0, no-op) adds a second, T1-only-ranked pool of extra
    candidates on top of the eps-lex shortlist -- the eps-lex shortlist finds
    good (T2,T3) tiebreaks among near-best-T1 samples, but the pool's raw
    best-T1 sample is sometimes still worse than a competing heuristic's T1;
    widening `num_sample` alone doesn't reliably surface it into the eps-band
    shortlist. Measured on `data/osm_test_small` (2858 instances,
    `outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt`): raising
    `num_sample` 100->400 and adding `topk_t1=100` moved "T1 not worse than
    ILS" 82.75%->95.31%, "T2 not worse" 44.2%->53.1%, lex-win 69.3%->86.4%,
    Pareto-dominance 26.2%->36.5%, at 0.144s/instance (still 5.6x faster than
    ILS's 0.81s/instance) -- see `logs/status_bm.md` for the full repro.

    `oracle_head` (default False, no-op) adds a second, independent candidate
    arm for variant P: an exact T1 oracle (`solvers.oracle.t1_oracle_argmin`,
    Held-Karp + partition DP over the class-1 arcs) yields a provably
    T1-optimal per-vehicle class-1 opening prefix, which is forced into the
    policy's Joint M x N decode; classes 2/3 are then completed with
    `oracle_r` sampled rollouts + 1 greedy, shortlisted by a T2-eps-band
    (all completions share the forced T1), local-searched, and the arm's own
    best is unioned with the pool arm's pick (`oracle_union_pick`). The
    union winner's T-vector is re-evaluated through the float64
    `SolutionEvaluator` before being returned. When the oracle is not
    applicable (class-1 arc count > `oracle_max_n1`, or no capacity-feasible
    partition exists), the head arm is skipped and the pool arm's answer is
    returned unchanged. Measured on `data/osm_test_small` (2858 instances,
    same ckpt as above, on top of NUM_SAMPLE=400 TOPK_T1=100): "T1 not worse
    than ILS" 95.31%->100%, T2 52.9%->86.0%, T3 55.9%->63.5%, lex-win
    86.4%->98.3%, Pareto 36.5%->57.7%, at ~3.4x ILS speed.
    """

    def __init__(self, pw, variant, device='cuda', shortlist=10, eps=0.005, topk_t1=0,
                 oracle_head=False, oracle_r=64, oracle_shortlist=8,
                 oracle_shortlist_eps=0.01, oracle_union_eps2=0.02,
                 oracle_tiebreak=True, oracle_max_n1=None):
        super().__init__(pw, variant, device=device)
        self.shortlist = shortlist
        self.eps = eps
        self.topk_t1 = topk_t1
        self.oracle_head = oracle_head
        self.oracle_r = oracle_r
        self.oracle_shortlist = oracle_shortlist
        self.oracle_shortlist_eps = oracle_shortlist_eps
        self.oracle_union_eps2 = oracle_union_eps2
        self.oracle_tiebreak = oracle_tiebreak
        if oracle_max_n1 is None:
            from solvers.oracle import MAX_N1_DEFAULT
            oracle_max_n1 = MAX_N1_DEFAULT
        self.oracle_max_n1 = oracle_max_n1
        from solvers.evaluator import SolutionEvaluator
        self.evaluator = SolutionEvaluator(variant=variant)

    def _local_search_one(self, vehicles, td0, M, adj, clss, demands, classes):
        """Run intra/inter local search on a decoded vehicles structure.

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
        if self.variant == 'P':
            for i, route in enumerate(trips):
                for p in classes:
                    route = intra_route_opt(route, adj, clss, variant='P', p=p)
                trips[i] = route
        else:
            trips = [intra_route_opt(r, adj, clss, variant='U') for r in trips]
        trips = inter_route_opt(trips, adj, demands, clss, capacity=1.0, variant=self.variant)

        new_vehicles = [[] for _ in range(M)]
        for m, route in zip(owner, trips):
            trip = route[1:-1]
            if len(trip):
                new_vehicles[m].append(trip)
        return self.evaluator._completion_times(new_vehicles, td0), new_vehicles

    @torch.inference_mode()
    def _decode_forced(self, forced, B, greedy=False):
        """Replicate the loaded instance B times, force `forced` flat Joint
        M x N actions as the decode opening, then let the policy complete by
        sampling (or argmax when `greedy`).

        Bypasses the policy's normal sampling entrypoint: runs the encoder
        once, then steps the env directly so arbitrary actions can be
        injected. Every forced action is checked against the env's
        `action_mask`; an infeasible one (an oracle/env mismatch -- never
        observed on the benchmark) aborts by returning None so the caller
        falls back to the pool arm.

        Returns (reward (B, P) float64 np, actions (B, S) int64 np) or None.
        """
        td = batchify(self.td, B) if B > 1 else self.td.clone()
        hidden, _ = self.policy.encoder(td)
        td, env, cache = self.policy.decoder.pre_decoder_hook(td, self.env, hidden)
        dev = td.device
        acts = []
        for a in forced:
            if not bool(td["action_mask"][0, a]):
                return None
            sel = torch.full((B,), int(a), dtype=torch.long, device=dev)
            td.set("action", sel)
            td = env.step(td)
            acts.append(sel)
        steps = 0
        while not td["done"].all():
            logits, mask = self.policy.decoder(td, cache)
            lf = logits.reshape(B, -1).masked_fill(~mask.reshape(B, -1), float('-inf'))
            if greedy:
                sel = lf.argmax(-1)
            else:
                sel = torch.multinomial(torch.softmax(lf, -1), 1).squeeze(1)
            td.set("action", sel)
            td = env.step(td)
            acts.append(sel)
            steps += 1
            if steps > 10000:
                raise RuntimeError("forced decode did not terminate")
        reward = env.get_reward(td).cpu().numpy().astype(np.float64)
        actions = torch.stack(acts, 1).cpu().numpy().astype(np.int64)
        return reward, actions

    def _oracle_head_arm(self, td0, M, N1, adj, clss, demands, classes):
        """Compute the oracle-head arm's own best candidate.

        Pipeline (see class docstring): exact T1 oracle -> forced-prefix
        decode (`oracle_r` sampled + 1 greedy completion) -> T2-eps-band
        shortlist -> local search (P1 convention: keep post-LS only if
        strictly lex-better) -> arm-internal pick via `t2_band_pick`.

        Returns (T_vec float64 (3,), vehicles) or (None, None) when the
        oracle is not applicable for this instance.
        """
        from solvers.oracle import forced_flat_actions, t1_oracle_argmin

        if self.variant != 'P':
            # The oracle's premise (class-1 arcs form a forced opening prefix)
            # only holds under variant P's precedence; decline otherwise.
            return None, None
        svc = td0["service_times"].cpu().numpy().astype(np.float64)
        res = t1_oracle_argmin(adj, svc, clss, demands, M,
                               tiebreak=self.oracle_tiebreak,
                               max_n1=self.oracle_max_n1)
        if not np.isfinite(res["T1"]) or res["parts"] is None:
            return None, None                       # oracle not applicable
        forced = forced_flat_actions(res["parts"], N1)

        pools = []                                  # (reward, actions, shortlist idxs)
        g = self._decode_forced(forced, 1, greedy=True)
        if g is None:
            return None, None                       # forced prefix infeasible
        pools.append((g[0], g[1], [0]))
        if self.oracle_r > 0:
            s_rew, s_act = self._decode_forced(forced, self.oracle_r, greedy=False)
            sl = shortlist_completions(s_rew, self.oracle_shortlist,
                                       eps=self.oracle_shortlist_eps)
            pools.append((s_rew, s_act, sl))

        cands = []                                  # (T_vec, vehicles)
        for rew, act, idxs in pools:
            for idx in idxs:
                idx = int(idx)
                pre_T = np.asarray(rew[idx], dtype=np.float64)
                pre_vehicles = self.evaluator.routes_from_joint_actions(
                    act[idx], M, N1)
                ls_T, ls_vehicles = self._local_search_one(
                    pre_vehicles, td0, M, adj, clss, demands, classes)
                if ls_T is not None and tuple(ls_T.tolist()) < tuple(pre_T.tolist()):
                    cands.append((np.asarray(ls_T, dtype=np.float64), ls_vehicles))
                else:
                    cands.append((pre_T, pre_vehicles))
        Ts = np.stack([c[0] for c in cands])
        j = t2_band_pick(Ts, self.oracle_union_eps2)
        return cands[j]

    def __call__(self, num_sample=100):
        td = batchify(self.td, num_sample)
        with torch.inference_mode():
            out = self.policy(td, env=self.env, decode_type='sampling')
        reward = out["reward"].cpu().numpy()                          # (num_sample, P)
        actions = out["actions"].cpu().numpy().astype(np.int64)       # (num_sample, dec_steps)

        keep = eps_lex_select(reward, eps=self.eps, shortlist=self.shortlist)
        if self.topk_t1 > 0:
            top = np.argsort(reward[:, 0])[:self.topk_t1]
            keep = np.array(sorted(set(keep.tolist()) | set(int(x) for x in top)),
                            dtype=keep.dtype)

        td0 = self.td[0]
        M = int(td0["num_vehicle"])
        N1 = td0["service_times"].shape[0]
        adj = td0["adj"].cpu().numpy().astype(np.float64)
        clss = td0["clss"].cpu().numpy().astype(np.int64)
        demands = td0["demand"].cpu().numpy().astype(np.float64)
        classes = [c for c in np.unique(clss) if c != 0]
        # SolutionEvaluator only lazily caches (adj/svc/clss) on first use, and
        # we call `_completion_times` directly (bypassing `__call__`, which is
        # what normally refreshes the cache) -- must refresh explicitly per
        # instance or later, differently-sized instances read stale arrays.
        self.evaluator._cache_td(td0)

        best_T, best_key, best_vehicles = None, None, None
        for idx in keep:
            pre_T = reward[idx]
            pre_vehicles = self.evaluator.routes_from_joint_actions(actions[idx], M, N1)
            ls_T, ls_vehicles = self._local_search_one(
                pre_vehicles, td0, M, adj, clss, demands, classes)
            # local search optimizes total deadhead (a proxy for T_k), so only
            # keep it if it's actually lex-better than the pre-LS rollout.
            if ls_T is not None and tuple(ls_T.tolist()) < tuple(pre_T.tolist()):
                cand_T, cand_vehicles = ls_T, ls_vehicles
            else:
                cand_T, cand_vehicles = pre_T, pre_vehicles
            key = tuple(cand_T.tolist())
            if best_key is None or key < best_key:
                best_key, best_T, best_vehicles = key, cand_T, cand_vehicles

        # ---- optional oracle-head arm (opt-in; never alters the pool arm) --
        if self.oracle_head:
            head_T, head_vehicles = self._oracle_head_arm(
                td0, M, N1, adj, clss, demands, classes)
            if head_T is not None:
                if oracle_union_pick(np.asarray(best_T, dtype=np.float64),
                                     head_T, self.oracle_union_eps2):
                    best_T, best_vehicles = head_T, head_vehicles
                # re-evaluate the union winner at float64 (same evaluator path
                # local search uses) so the returned T-vector -- and the T1
                # optimality guarantee -- hold at full precision, not just the
                # decode-time fp32 reward's.
                best_T = self.evaluator._completion_times(best_vehicles, td0)

        # expose the winning solution structure (not just its T-vector) for
        # downstream consumers, e.g. the RL-warm-started ILS.
        self._last_best_vehicles = best_vehicles
        self._last_best_action = vehicles_to_action(best_vehicles)
        return np.asarray(best_T, dtype=np.float64)[None, :]           # (1, P)


class RLWarmILSHCARP(RLHybHCARP):
    """Three-tier funnel: RL sampling + eps-lex shortlist + local search
    (RLHybHCARP), then a SHORT Iterated Local Search warm-started from the
    hybrid's winning solution instead of ILS's usual random construction.

    The hybrid winner's per-vehicle trips are flattened to the metaheuristics'
    0-separated arc-order format (`vehicles_to_action`) and passed to
    `ILSHCARP(init_action=...)` with a reduced iteration budget (default 50 vs
    ILS's usual 200). Note the two representations are evaluated differently:
    ILS's evaluator (`SolutionEvaluator._build_P` for variant P) re-derives the
    vehicle assignment from the arc order alone, so the warm start's T-vector
    can differ from the hybrid's joint-decoded T-vector. The final result is
    therefore the lex-best of (hybrid T, ILS T) — provably no worse than
    `RLHybHCARP` alone.
    """

    def __init__(self, pw, variant, device='cuda', shortlist=10, eps=0.005,
                 topk_t1=0, ils_iters=50, ils_strength=3, seed=None):
        super().__init__(pw, variant, device=device, shortlist=shortlist, eps=eps,
                          topk_t1=topk_t1)
        self.ils_iters = ils_iters
        self.ils_strength = ils_strength
        self.seed = seed

    def _load_data(self, *data):
        super()._load_data(*data)
        self._instance_data = data          # kept for the ILS refiner

    def __call__(self, num_sample=100):
        from solvers.meta import ILSHCARP

        rl_T = super().__call__(num_sample=num_sample)              # (1, P)
        init_action = self._last_best_action
        if init_action.size == 0:
            return rl_T

        ils = ILSHCARP(strength=self.ils_strength, accept_mode='best')
        ils._load_data(*self._instance_data)
        ils_T = ils(max_iter=self.ils_iters, variant=self.variant,
                    init_action=init_action, seed=self.seed)        # (1, P)
        self._last_ils_action = ils._last_best_action

        if tuple(ils_T[0].tolist()) < tuple(rl_T[0].tolist()):
            return ils_T
        return rl_T


def parse_args():
    parser = argparse.ArgumentParser(description="RLHCARP")
    parser.add_argument('--seed', type=int, default=6868)
    parser.add_argument('--variant', default='P', choices=['P', 'U'])
    parser.add_argument('--num_sample', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.005,
                        help='relative T1 tolerance for eps-lex selection (default 0.5%%)')
    # single-instance log mode
    parser.add_argument('--ckpt', default=None, help='checkpoint (.ckpt); enables log mode')
    parser.add_argument('--file', default=None, help='.npz instance; enables log mode')
    parser.add_argument('--vehicles', type=int, default=None, help='override fleet size')
    parser.add_argument('--log', default='outputs/rl_route_log.txt')
    parser.add_argument('--solution', default=None)
    # batch benchmark mode
    parser.add_argument('--cpkt', default=None, help='checkpoint for batch mode')
    parser.add_argument('--path', default='data/5m', help='directory of .npz instances')
    parser.add_argument('--M', type=int, default=None)
    parser.add_argument('--hybrid', action='store_true',
                        help='batch mode: two-tier funnel (eps-lex shortlist + local search)')
    parser.add_argument('--shortlist', type=int, default=10,
                        help='hybrid mode: #candidates that get local search')
    parser.add_argument('--topk_t1', type=int, default=0,
                        help='hybrid mode: extra #candidates picked by raw T1 rank '
                             '(on top of the eps-lex shortlist), local-searched too. '
                             '0 = disabled (default, no behavior change)')
    parser.add_argument('--oracle_head', action='store_true',
                        help='hybrid mode: add the exact-T1-oracle forced-prefix arm '
                             '(variant P) and union it with the pool arm '
                             '(default: off, no behavior change)')
    parser.add_argument('--oracle_r', type=int, default=64,
                        help='oracle_head mode: #sampled completions per forced prefix '
                             '(+1 greedy; default 64)')
    parser.add_argument('--oracle_shortlist', type=int, default=8,
                        help='oracle_head mode: #completions that get local search '
                             '(default 8)')
    parser.add_argument('--oracle_shortlist_eps', type=float, default=0.01,
                        help='oracle_head mode: relative T2 eps-band for completion '
                             'shortlisting (default 0.01)')
    parser.add_argument('--oracle_union_eps2', type=float, default=0.02,
                        help='oracle_head mode: relative T2 eps-band for the arm-internal '
                             'pick and the P1-vs-head union (default 0.02)')
    parser.add_argument('--oracle_tiebreak', type=int, default=1,
                        help='oracle_head mode: 1 = exact lexicographic tiebreak among '
                             'T1-optimal prefixes (default), 0 = plain argmin')
    parser.add_argument('--warm_ils', action='store_true',
                        help='batch mode: hybrid funnel + short ILS refinement '
                             'warm-started from the hybrid winner (implies --hybrid)')
    parser.add_argument('--ils_iters', type=int, default=50,
                        help='warm_ils mode: ILS iteration budget (default 50)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = args.ckpt or args.cpkt

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
        files = sorted(glob(args.path + '/*/*.npz'))
        if args.warm_ils:
            al = RLWarmILSHCARP(ckpt_path, args.variant, shortlist=args.shortlist,
                                eps=args.eps, topk_t1=args.topk_t1,
                                ils_iters=args.ils_iters, seed=args.seed)
        elif args.hybrid:
            al = RLHybHCARP(ckpt_path, args.variant, shortlist=args.shortlist,
                            eps=args.eps, topk_t1=args.topk_t1,
                            oracle_head=args.oracle_head, oracle_r=args.oracle_r,
                            oracle_shortlist=args.oracle_shortlist,
                            oracle_shortlist_eps=args.oracle_shortlist_eps,
                            oracle_union_eps2=args.oracle_union_eps2,
                            oracle_tiebreak=bool(args.oracle_tiebreak))
        else:
            al = RLHCARP(ckpt_path, args.variant)
        for f in files:
            for i, data in enumerate(iter_instances(f, M=args.M)):
                al._load_data(*data)
                label = f if i == 0 else f"{f}[{i}]"
                t1 = time()
                if args.hybrid or args.warm_ils:
                    result = al(num_sample=args.num_sample)
                else:
                    result = al(num_sample=args.num_sample, eps=args.eps)
                print(label, ':::', result, ':::', time() - t1)