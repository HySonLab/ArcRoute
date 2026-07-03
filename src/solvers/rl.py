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
    """

    def __init__(self, pw, variant, device='cuda', shortlist=10, eps=0.005, topk_t1=0):
        super().__init__(pw, variant, device=device)
        self.shortlist = shortlist
        self.eps = eps
        self.topk_t1 = topk_t1
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
                            eps=args.eps, topk_t1=args.topk_t1)
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