"""Measure Scheduler gap across two angles.

  Angle 1 — Assignment gap (Scheduler vs ILS direct assignment)
    ILS  → best routes → T_ILS          (ILS assigns vehicles directly)
    Flatten ILS arc order → Scheduler   → T_ils_sched
    Gap_A = T_ils_sched - T_ILS

    Interpretation: how much quality does the Scheduler lose when it has to
    re-assign vehicles from a good arc order (ILS's), vs ILS doing it directly?

  Angle 2 — Ordering sensitivity (policy vs ILS arc order, via same Scheduler)
    Policy arc order  → Scheduler → T_policy_sched
    ILS arc order     → Scheduler → T_ils_sched    (reused from Angle 1)
    Random within-class permutations → Scheduler → T_rand_best

    Gap_order = T_policy_sched - T_ils_sched   (policy ordering vs ILS ordering)
    Gap_rand  = T_policy_sched - T_rand_best   (room for ordering improvement)

    Interpretation:
      Gap_A >> Gap_order → Scheduler assignment is the bottleneck
      Gap_order >> Gap_A → Policy arc ordering quality is the bottleneck
      Gap_rand  ≈ 0      → Policy ordering already near-locally-optimal for Scheduler

Usage:
    uv run python scripts/measure_scheduler_gap.py \\
        --ckpt  outputs/checkpoints/curriculum_small/epoch=098.ckpt \\
        --ood_dir data/ood/osm_cityA/40 \\
        --n_instances 20 \\
        --n_ils_samples 30 \\
        --n_policy_samples 64 \\
        --n_rand_perms 500 \\
        --vehicles 2
"""
import sys, os, glob, argparse, warnings
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.ops import import_instance, batchify
from utils.nb_utils import deserialize_tours_batch
from utils.local_search import ls
from solvers.meta import InsertCheapestHCARP
from solvers.cal_reward import get_Ts
from solvers.scheduler import Scheduler

warnings.filterwarnings('ignore')


# ── helpers ─────────────────────────────────────────────────────────────────

def lex_score(T):
    """Scalar for lex comparison: T[0] dominates."""
    return T[0] * 1e6 + T[1] * 1e3 + T[2]


def solver_td(solver):
    """Build Scheduler-compatible td dict from InsertCheapestHCARP.vars."""
    v = solver.vars
    return {
        'adj':           v['adj'],
        'service_times': v['service_time'],   # ILS uses 'service_time' key
        'clss':          v['clss'],
        'demand':        v['demand'],
    }


def flatten_routes(routes):
    """[0,a1,a2,...,0] routes → 1-D arc order, 0s removed."""
    return np.concatenate([np.asarray(r)[1:-1] for r in routes
                           if len(r) > 2]).astype(np.int64)


def run_random_perms(arc_order_no_depot, td, scheduler, M, n_perms, clss_arr):
    """Shuffle within-class arc order n_perms times, return best T (lex)."""
    cls = np.asarray(clss_arr).ravel()
    # group arc indices by class (using the class of each arc in the ordering)
    groups = {}
    for arc in arc_order_no_depot:
        c = int(cls[arc])
        groups.setdefault(c, []).append(arc)

    best_T, best_score = None, np.inf
    for _ in range(n_perms):
        perm = np.concatenate([
            np.random.permutation(groups[c])
            for c in sorted(groups.keys())
        ]).astype(np.int64)
        _, T = scheduler(perm, td, M=M)
        sc = lex_score(T)
        if sc < best_score:
            best_T, best_score = T.copy(), sc
    return best_T


# ── Angle 1: ILS direct vs Scheduler(ILS arc order) ─────────────────────────

def angle1_instance(npz_file, M, n_ils_samples, variant, scheduler):
    solver = InsertCheapestHCARP()
    solver.import_instance(npz_file, M=M)
    M_used = solver.nv

    raw = [solver.get_once(solver.M, solver.clss) for _ in range(n_ils_samples)]
    raw = [a for a in raw if a is not None]
    if not raw:
        return None

    tours_batch = ls(solver.vars, variant=variant, actions=raw)
    actions_arr = deserialize_tours_batch(tours_batch, solver.nseq)
    Ts_all   = get_Ts(solver.vars, actions=actions_arr)
    best_idx = lex_score(Ts_all.T).argmin() if Ts_all.ndim == 2 else 0
    T_ils    = Ts_all[best_idx]
    best_routes = tours_batch[best_idx]

    arc_order_ils = flatten_routes(best_routes)
    td = solver_td(solver)
    _, T_ils_sched = scheduler(arc_order_ils, td, M=M_used)

    return {
        'M':           M_used,
        'n_arcs':      solver.nseq,
        'T_ils':       T_ils,
        'T_ils_sched': T_ils_sched,
        'gap_A':       T_ils_sched - T_ils,
        'arc_order_ils': arc_order_ils,
        'solver':      solver,
    }


# ── Angle 2: Policy ordering vs ILS ordering (both via Scheduler) ─────────────

def load_policy(ckpt_path, device):
    from policy.policy import AttentionModelPolicy
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp   = ckpt.get('hyper_parameters', {})
    policy = AttentionModelPolicy(
        embed_dim=hp.get('embed_dim', 128),
        num_encoder_layers=hp.get('num_encoder_layers', 12),
        num_heads=hp.get('num_heads', 8),
    )
    sd = {k[len('policy.'):]: v for k, v in ckpt['state_dict'].items()
          if k.startswith('policy.')}
    policy.load_state_dict(sd, strict=True)
    return policy.to(device).eval()


def build_env(ckpt_path, variant):
    from env.env import CARPEnv
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp   = ckpt.get('hyper_parameters', {})
    return CARPEnv(
        num_loc=hp.get('num_loc', 40),
        num_arc=hp.get('num_arc', 80),
        variant=variant,
        reward_mode='vector',
    )


def build_policy_td(npz_file, M, env, device):
    from tensordict import TensorDict
    dms, P, M_inst, demands, clss, s, d, _ = import_instance(npz_file, M=M)
    M_use = M if M is not None else M_inst
    td = TensorDict({
        'clss':            torch.tensor(clss[None, :],   dtype=torch.int64),
        'demands':         torch.tensor(demands[None, :], dtype=torch.float32),
        'service_times':   torch.tensor(s[None, :],       dtype=torch.float32),
        'traversal_times': torch.tensor(d[None, :],       dtype=torch.float32),
        'adj':             torch.tensor(dms[None, :],     dtype=torch.float32),
        'num_vehicle':     torch.tensor([[M_use]],        dtype=torch.int64),
    }, batch_size=[1])
    return env.reset(td).to(device), M_use


def angle2_instance(npz_file, angle1_result, policy, env, scheduler,
                    n_policy_samples, n_rand_perms, device):
    if angle1_result is None:
        return None

    M      = angle1_result['M']
    td, _  = build_policy_td(npz_file, M, env, device)
    td_k   = batchify(td, n_policy_samples)

    with torch.inference_mode():
        out = policy(td_k, env=env, phase='test')

    Ts_all = out['reward'].cpu().numpy()            # (K, P)
    scores = np.array([lex_score(Ts_all[i]) for i in range(len(Ts_all))])
    best_k = int(scores.argmin())
    T_policy_sched_from_reward = Ts_all[best_k]

    # Get arc ordering from best rollout, run through Scheduler explicitly
    best_action = out['actions'][best_k].cpu().numpy().astype(np.int64)
    td_single = {k: v for k, v in angle1_result['solver'].vars.items()
                 if k != 'nv'}
    td_single['service_times'] = td_single.pop('service_time')

    _, T_policy_sched = scheduler(best_action, td_single, M=M)

    # Random within-class permutations (sensitivity test)
    arc_order_policy = best_action[best_action != 0]
    clss_arr = angle1_result['solver'].vars['clss']
    T_rand_best = run_random_perms(arc_order_policy, td_single, scheduler,
                                   M, n_rand_perms, clss_arr)

    T_ils_sched = angle1_result['T_ils_sched']
    T_ils       = angle1_result['T_ils']

    return {
        'T_policy_sched':  T_policy_sched,
        'T_ils_sched':     T_ils_sched,
        'T_ils':           T_ils,
        'T_rand_best':     T_rand_best,
        'gap_A':           T_ils_sched - T_ils,           # Scheduler assignment loss
        'gap_order':       T_policy_sched - T_ils_sched,  # policy ordering vs ILS ordering
        'gap_rand':        T_policy_sched - T_rand_best,  # room from random search
        'total_gap':       T_policy_sched - T_ils,        # total vs ILS ground truth
    }


# ── reporting ────────────────────────────────────────────────────────────────

def report_angle1(results):
    print('\n' + '=' * 65)
    print('  ANGLE 1 — Assignment gap  (Scheduler vs ILS direct)')
    print('  Question: When given ILS\'s arc order, does Scheduler')
    print('            recover ILS\'s vehicle assignment quality?')
    print('=' * 65)
    P = len(results[0]['T_ils'])
    for k in range(P):
        ils      = np.mean([r['T_ils'][k]       for r in results])
        ils_sch  = np.mean([r['T_ils_sched'][k]  for r in results])
        gap      = np.mean([r['gap_A'][k]        for r in results])
        pct      = gap / max(ils, 1e-9) * 100
        print(f'  T{k+1}: ILS={ils:.4f}  Scheduler(ILS order)={ils_sch:.4f}'
              f'  gap={gap:+.4f} ({pct:+.1f}%)')
    print()
    lex_better = sum(lex_score(r['T_ils_sched']) <= lex_score(r['T_ils'])
                     for r in results)
    print(f'  Scheduler(ILS order) lex-better-or-equal than ILS: '
          f'{lex_better}/{len(results)} instances')
    print()


def report_angle2(results):
    print('=' * 65)
    print('  ANGLE 2 — Ordering sensitivity')
    print('  Decomposition: Policy vs ILS arc order (same Scheduler)')
    print('=' * 65)
    P = len(results[0]['T_ils'])

    rows = [
        ('ILS direct',          'T_ils',         '(ground truth)'),
        ('Scheduler(ILS order)','T_ils_sched',   ''),
        ('Scheduler(Policy)',   'T_policy_sched',''),
        ('Sched(rand-best)',    'T_rand_best',   f'(best of random perms)'),
    ]

    print(f'  {"":30s}  {"T1":>8}  {"T2":>8}  {"T3":>8}')
    print('  ' + '-' * 59)
    for label, key, note in rows:
        means = [np.mean([r[key][k] for r in results]) for k in range(P)]
        vals  = '  '.join(f'{v:8.4f}' for v in means)
        print(f'  {label:30s}  {vals}  {note}')

    print()
    print('  Gaps (mean across instances):')
    labels = [
        ('Gap_A  (Scheduler assign loss)', 'gap_A'),
        ('Gap_order (Policy vs ILS order)', 'gap_order'),
        ('Gap_rand  (room via rand perm)',  'gap_rand'),
        ('Gap_total (Policy vs ILS direct)','total_gap'),
    ]
    for label, key in labels:
        gaps = [np.mean([r[key][k] for r in results]) for k in range(P)]
        vals = '  '.join(f'{v:+8.4f}' for v in gaps)
        print(f'  {label:40s}  {vals}')

    print()
    print('  Verdict:')
    gap_A     = np.mean([r['gap_A'][0]     for r in results])
    gap_order = np.mean([r['gap_order'][0] for r in results])
    gap_rand  = np.mean([r['gap_rand'][0]  for r in results])

    if abs(gap_A) > abs(gap_order) * 2:
        print('  → Scheduler assignment is the PRIMARY bottleneck (Gap_A >> Gap_order)')
        print('    Improving policy ordering alone won\'t help much.')
        print('    Consider improving the Scheduler\'s _assign / _build_P heuristic.')
    elif abs(gap_order) > abs(gap_A) * 2:
        print('  → Policy arc ORDERING is the PRIMARY bottleneck (Gap_order >> Gap_A)')
        print('    Scheduler assignment quality is fine.')
        print('    Focus on improving policy (context enrichment, Tier 1).')
    else:
        print('  → Both ordering AND Scheduler assignment contribute equally.')

    if gap_rand > 0.01:
        print(f'  → Gap_rand T1={gap_rand:.4f}: random permutations find better')
        print('    orderings — policy within-class arc choice has room to improve.')
    else:
        print('  → Gap_rand ≈ 0: policy arc ordering already near-locally-optimal.')
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',             required=True)
    parser.add_argument('--ood_dir',          default='data/ood/osm_cityA/40')
    parser.add_argument('--n_instances',      type=int, default=20)
    parser.add_argument('--n_ils_samples',    type=int, default=30)
    parser.add_argument('--n_policy_samples', type=int, default=64)
    parser.add_argument('--n_rand_perms',     type=int, default=500)
    parser.add_argument('--vehicles',         type=int, default=None)
    parser.add_argument('--variant',          default='P', choices=['P', 'U'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    npz_files = sorted(glob.glob(os.path.join(args.ood_dir, '*.npz')))
    if not npz_files:
        print(f'No .npz files found in {args.ood_dir}'); return
    npz_files = npz_files[:args.n_instances]
    print(f'Instances : {len(npz_files)} from {args.ood_dir}')
    print(f'ILS samples: {args.n_ils_samples} | '
          f'Policy samples: {args.n_policy_samples} | '
          f'Rand perms: {args.n_rand_perms}')

    scheduler = Scheduler(variant=args.variant)

    # ── Angle 1 ──────────────────────────────────────────────────────────────
    print('\nRunning Angle 1 (ILS + Scheduler) ...')
    a1_results = []
    for i, f in enumerate(npz_files):
        print(f'  [{i+1:2d}/{len(npz_files)}] {os.path.basename(f)}', end=' ', flush=True)
        r = angle1_instance(f, args.vehicles, args.n_ils_samples,
                            args.variant, scheduler)
        if r is None:
            print('SKIP (no feasible ILS)')
        else:
            print(f'T_ILS=({r["T_ils"][0]:.3f},{r["T_ils"][1]:.3f},'
                  f'{r["T_ils"][2]:.3f})  '
                  f'T_sched=({r["T_ils_sched"][0]:.3f},{r["T_ils_sched"][1]:.3f},'
                  f'{r["T_ils_sched"][2]:.3f})')
            a1_results.append(r)

    if not a1_results:
        print('All ILS runs failed.'); return

    report_angle1(a1_results)

    # ── Angle 2 ──────────────────────────────────────────────────────────────
    print('Running Angle 2 (Policy + random perms) ...')
    policy = load_policy(args.ckpt, device)
    env    = build_env(args.ckpt, args.variant)

    a2_results = []
    for i, (f, a1r) in enumerate(zip(npz_files[:len(a1_results)], a1_results)):
        print(f'  [{i+1:2d}/{len(a1_results)}] {os.path.basename(f)}',
              end=' ', flush=True)
        r = angle2_instance(f, a1r, policy, env, scheduler,
                            args.n_policy_samples, args.n_rand_perms, device)
        if r is None:
            print('SKIP')
        else:
            print(f'T_policy=({r["T_policy_sched"][0]:.3f},'
                  f'{r["T_policy_sched"][1]:.3f},{r["T_policy_sched"][2]:.3f})'
                  f'  rand_best=({r["T_rand_best"][0]:.3f},'
                  f'{r["T_rand_best"][1]:.3f},{r["T_rand_best"][2]:.3f})')
            a2_results.append(r)

    if a2_results:
        report_angle2(a2_results)


if __name__ == '__main__':
    main()
