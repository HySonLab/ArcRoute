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

from utils.ops import batchify, gen_tours, import_instance, print_route_log, save_sol


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
        self.policy = policy.to(device).eval()
        self.env = CARPEnv(variant=variant, reward_mode="vector")

    def import_instance(self, f, M=None):
        dms, P, veh, demands, clss, s, d, _ = import_instance(f, M=M)
        M_val = M if M is not None else len(veh)
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

    def __call__(self, num_sample=100):
        td = batchify(self.td, num_sample)
        with torch.inference_mode():
            out = self.policy(td, env=self.env, phase='test')
        reward = out["reward"].cpu().numpy()          # (num_sample, P)
        idx = int(np.lexsort(reward[:, ::-1].T)[0])  # lex best (lowest T1 primary)
        return reward[idx:idx+1]                      # (1, P)


def parse_args():
    parser = argparse.ArgumentParser(description="RLHCARP")
    parser.add_argument('--seed', type=int, default=6868)
    parser.add_argument('--variant', default='P', choices=['P', 'U'])
    parser.add_argument('--num_sample', type=int, default=100)
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
            out = policy(td_k, env=env, phase="test")
        Ts_all = out["reward"].cpu().numpy()
        idx = int(np.lexsort((Ts_all[:, 2], Ts_all[:, 1], Ts_all[:, 0]))[0])
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
        al = RLHCARP(ckpt_path, args.variant)
        for f in files:
            al.import_instance(f, M=args.M)
            t1 = time()
            print(f, ':::', al(num_sample=args.num_sample), ':::', time() - t1)