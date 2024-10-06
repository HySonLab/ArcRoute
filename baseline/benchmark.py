import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.hr import InsertCheapestHCARP
from common.local_search import local_search
from common.cal_reward import get_Ts
from rl.ppo import PPO
from common.ops import import_instance, batchify

import glob
from tqdm import tqdm
import numpy as np
import torch


if __name__ == "__main__":
    torch.manual_seed(6868)
    np.random.seed(6868)
    files = glob.glob('/home/project/testing_data/medium/*.npz')
    files = sorted(files, key=lambda x : int(x.split('/')[-1].split('_')[0]))[:10]
    results = {
        "hr": [],
        "rl": []
    }
    
    # Heuristics
    al = InsertCheapestHCARP()
    for f in tqdm(files):
        al.import_instance(f)
        routes = al(merge_tour=True)
        vars = {
            'adj': al.dms[None],
            'service_time': al.s[None],
            'clss': al.clss[None],
            'demand': al.demands[None]
        }
        tours = local_search(vars, actions=routes[None], variant='U')
        r = get_Ts(vars, tours_batch=tours)
        results['hr'].append(r[0])
    results['hr'] = np.array(results['hr'])

    # print(results['hr'])
    # exit()

        
    # Hybrid Reinforce
    model = PPO.load_from_checkpoint('/home/project/cpkts/cl1/epoch=025.ckpt')
    policy, env = model.policy.cuda(), model.env
    for f in tqdm(files):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        td = env.reset(batch_size=1)
        td['clss'] = torch.tensor(clss[None, :], dtype=torch.int64)
        td['demand'] = torch.tensor(demands[1:][None, :], dtype=torch.float32)
        td['service_time'] = torch.tensor(s[None, :], dtype=torch.float32)
        td['traveling_time'] = torch.tensor(d[None, :], dtype=torch.float32)
        td['adj'] = torch.tensor(dms[None, :], dtype=torch.float32)
        td = td.cuda()

        # td = batchify(td, 10)
        # with torch.no_grad():
        #     out = policy(td, env=env, decode_type='sampling', return_actions=True)
        #     obj = env.get_objective(td, out['actions'])
        #     idx = obj[:, 0].argmin()
            # results['rl'].append(obj[idx].numpy())

        with torch.no_grad():
            out = policy(td, env=env, decode_type='greedy', return_actions=True)
            obj = env.get_objective(td, out['actions'])
            results['rl'].append(obj[0])
    
    results['rl'] = np.array(results['rl'])

    # # print(results['rl'])
    # # exit()
    
    r1_rl = results['rl'][:, 0]
    r1_hr = results['hr'][:, 0]
    
    print("Gap%: hr - rl:", ((r1_hr - r1_rl)/(r1_rl+1e-8) * 100).mean())