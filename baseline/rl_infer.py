import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.ppo import PPO
import torch
import numpy as np
from common.ops import import_instance, batchify
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO model")

    # Add arguments
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file (.npz)')

    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(6868)
    np.random.seed(6868)
    args = parse_args()
    
    # Load model from checkpoint
    model = PPO.load_from_checkpoint(args.checkpoint_path)
    policy, env = model.policy.cuda(), model.env

    # Import data from .npz file
    dms, P, M, demands, clss, s, d, edge_indxs = import_instance(args.data_path)

    # Prepare the environment
    td = env.reset(batch_size=1)
    td['clss'] = torch.tensor(clss[None, :], dtype=torch.int64)
    td['demand'] = torch.tensor(demands[1:][None, :], dtype=torch.float32)
    td['service_time'] = torch.tensor(s[None, :], dtype=torch.float32)
    td['traveling_time'] = torch.tensor(d[None, :], dtype=torch.float32)
    td['adj'] = torch.tensor(dms[None, :], dtype=torch.float32)
    td = td.cuda()

    # Run the model with no gradients for inference
    with torch.no_grad():
        out = policy(td.clone(), env=env, decode_type='greedy', return_actions=True)
    
    # Output results
    print(out['actions'])
    print(env.get_objective(td, out['actions']))