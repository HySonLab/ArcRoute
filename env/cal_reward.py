import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
from common.ops import run_parallel, gen_tours_batch, setup_vars
import numpy as np
import torch

def get_reward(td, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
    bs = len(tours_batch)
    adjs, service_times, clsss = setup_vars(td)
    reward1 = run_parallel(reward_ins, [1]*bs, service_times, adjs, clsss, tours_batch)
    # reward2 = run_parallel(reward_ins, [2]*bs, service_times, adjs, clsss, tours_batch)
    # reward3 = run_parallel(reward_ins, [3]*bs, service_times, adjs, clsss, tours_batch)
    return reward1

def get_Ts(td, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
    bs = len(tours_batch)
    adjs, service_times, clsss = setup_vars(td)
    reward1 = run_parallel(reward_ins, [1]*bs, service_times, adjs, clsss, tours_batch)
    reward2 = run_parallel(reward_ins, [2]*bs, service_times, adjs, clsss, tours_batch)
    reward3 = run_parallel(reward_ins, [3]*bs, service_times, adjs, clsss, tours_batch)
    return [torch.tensor(reward1), torch.tensor(reward2), torch.tensor(reward3)]

@nb.njit(nb.float32(nb.int32, nb.float32[:], nb.float32[:, :], nb.int32[:], nb.int32[:, :]), nogil=True)
def reward_ins(k, service_time, adj, clss, tours):
    r = 0.0
    for tour in tours:
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 0:
            continue
        candidate = tour[:pos[-1] + 1]
        length = sum([service_time[i] for i in candidate[1:]]) + sum([adj[i,j] for i, j in zip(candidate[:-1],candidate[1:])])  
        r = max(r, length)
    return r + 1000*max(0, len(tours) - 2)
