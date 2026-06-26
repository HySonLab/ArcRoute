import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solvers.meta import ILSHCARP
from time import time
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ILSHCARP")

    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--variant', type=str, default='P', help='Environment variant')
    parser.add_argument('--M', type=int, default=None, help='override fleet size at eval (Phase 3: M is a solve-time param)')
    parser.add_argument('--num_init_sample', type=int, default=5, help='number of constructive seeds for the initial solution')
    parser.add_argument('--max_iter', type=int, default=200, help='number of ILS iterations')
    parser.add_argument('--strength', type=int, default=3, help='perturbation strength (number of arcs relocated)')
    parser.add_argument('--accept_mode', type=str, default='best', choices=['best', 'sa'], help="acceptance criterion: 'best' or 'sa'")
    parser.add_argument('--path', type=str, default='/usr/local/rsa/ArcRoute/data/instances', help='path to instances')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    files = sorted(glob(args.path + '/*/*.npz'))

    al = ILSHCARP(strength=args.strength, accept_mode=args.accept_mode)
    for f in files:
        al.import_instance(f, M=args.M)
        t1 = time()
        print(f, ':::',
              al(max_iter=args.max_iter, variant=args.variant,
                 num_init_sample=args.num_init_sample, seed=args.seed),
              ':::', time() - t1)
