import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta import ACOHCARP
from time import time
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ACOHCARP")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--variant', type=str, default='U', help='Environment variant')
    parser.add_argument('--n_ant', type=int, default=50, help='num epoch')
    parser.add_argument('--path', type=str, default='data/5m60', help='path to instances')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    files = sorted(glob(args.path + '/*/*.npz'))
    
    al = ACOHCARP(n_ant=args.n_ant) # ACO
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::', al(n_epoch=args.max_epoch, variant=args.variant),':::', time() - t1)