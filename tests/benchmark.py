import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.hr import InsertCheapestHCARP
from envs.local_search import local_search
from envs.cal_reward import get_Ts
from models.ppo import PPO
from common.common import import_instance

from rl4co.utils.ops import batchify
import glob
from tqdm import tqdm
import numpy as np
import torch


if __name__ == "__main__":
    files = glob.glob('/home/project/ArcRoute/data/instances/15/*.npz')
    files = sorted(files, key=lambda x : int(x.split('/')[-1].split('_')[0]))
    results = {
        "hr": [],
        "rl": []
    }
    
    # Heuristics
    al = InsertCheapestHCARP()
    for f in tqdm(files):
        al.import_instance(f)
        routes = al(merge_tour=True)
        tours = local_search([al.dms[None],al.s[None],al.clss[None]], actions=[routes])
        r = get_Ts([al.dms[None], al.s[None], al.clss[None]], tours_batch=tours)
        results['hr'].append(r[0])
        
        
    # Hybrid Reinforce
    model = PPO.load_from_checkpoint('/home/project/checkpoints/cl1_old/best.ckpt')
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

        td = batchify(td, 100)
        with torch.no_grad():
            out = policy(td, env=env, decode_type='sampling', return_actions=True)
            obj = env.get_objective(td, out['actions'])
            idx = obj[:, 0].argmin()
            results['rl'].append(obj[idx].numpy())
            
    results['rl'] = np.array(results['rl'])
    results['hr'] = np.array(results['hr'])
    
    r1_rl = results['rl'][:, 0]
    r1_hr = results['hr'][:, 0]
    
    print("Gap%: hr - rl", ((r1_hr - r1_rl)/r1_rl * 100).mean())