import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.ppo import PPO
import torch
import numpy as np
from common.ops import import_instance, batchify

if __name__ == "__main__":
    torch.manual_seed(6868)
    np.random.seed(6868)
    model = PPO.load_from_checkpoint("/home/project/cpkts/cl1/epoch=016.ckpt")
    policy = model.policy.cuda()
    env = model.env

    # Import data from .npz file
    dms, P, M, demands, clss, s, d, edge_indxs = import_instance("/home/project/testing_data/small/1_32.npz")

    # # Prepare the environment
    td = env.reset(batch_size=1)
    # print(td)
    print(td['adj'].min(), td['adj'].max())
    td['clss'] = torch.tensor(clss[None, :], dtype=torch.int64)
    td['demand'] = torch.tensor(demands[1:][None, :], dtype=torch.float32)
    td['service_time'] = torch.tensor(s[None, :], dtype=torch.float32)
    td['traveling_time'] = torch.tensor(d[None, :], dtype=torch.float32)
    td['adj'] = torch.tensor(dms[None, :], dtype=torch.float32)
    print('-------')
    print(td['adj'].min(), td['adj'].max())
    td = td.cuda()

    # td = batchify(td, 10)
    with torch.no_grad():
        out = policy(td, env=env, decode_type='greedy', return_actions=True)

    print(env.get_objective(td, actions=out['actions']))

