import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.ppo import PPO
import torch
import numpy as np
from common.ops import import_instance, batchify
from glob import glob
from time import time
import argparse

class RLHCARP:
    def __init__(self, pw, device='cuda'):
        model = PPO.load_from_checkpoint(pw)
        self.device = device
        self.policy = model.policy.to(device)
        self.env = model.env
        
    
    def import_instance(self, f):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        td = self.env.reset(batch_size=1)
        td['clss'] = torch.tensor(clss[None, :], dtype=torch.int64)
        td['demand'] = torch.tensor(demands[1:][None, :], dtype=torch.float32)
        td['service_time'] = torch.tensor(s[None, :], dtype=torch.float32)
        td['traveling_time'] = torch.tensor(d[None, :], dtype=torch.float32)
        td['adj'] = torch.tensor(dms[None, :], dtype=torch.float32)
        td = self.env.reset(td)
        self.td = td.to(self.device)

    def __call__(self, num_sample=100):
        # with torch.inference_mode():
        #     out = self.policy(self.td, env=self.env, decode_type='greedy', return_actions=True)
        #     obj = self.env.get_objective(self.td, out['actions'])

        td = batchify(self.td, num_sample)
        with torch.inference_mode():
            out = self.policy(td, env=self.env, decode_type='sampling', return_actions=True)
            obj = self.env.get_objective(td, out['actions'])
            idx = obj[:, 0].argmin()
            obj = obj[idx]
        return obj


def parse_args():
    parser = argparse.ArgumentParser(description="RLHCARP")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--variant', type=str, default='P', help='Environment variant')
    parser.add_argument('--num_sample', type=int, default=500, help='num_sample')
    parser.add_argument('--cpkt', type=str, default='/usr/local/rsa/cpkts/bestP_20_2.ckpt', help='cpkt')
    parser.add_argument('--path', type=str, default='/usr/local/rsa/ArcRoute/data/instances', help='path to instances')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(glob(args.path + '/*/*.npz'))
    al = RLHCARP(args.cpkt)
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::', al(num_sample=args.num_sample),':::', time() - t1)