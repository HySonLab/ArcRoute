import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
from common.ops import run_parallel, convert_vars_np
from common.nb_utils import gen_tours_batch, calc_length


def get_reward(vars, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
    if not isinstance(vars, dict):
        vars = convert_vars_np(vars)
     
    reward1 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 1)
    # reward2 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 2)
    # reward3 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 3)
    # return np.float32([reward1, reward2, reward3]).T @ np.float32([1e2, 1e0, 1e-2])
    return reward1

def get_Ts(vars, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
    reward1 = run_parallel(reward_ins, tours_batch, adj=vars['adj'], service=vars['service_time'], clss=vars['clss'], k = 1)
    reward2 = run_parallel(reward_ins, tours_batch, adj=vars['adj'], service=vars['service_time'], clss=vars['clss'], k = 2)
    reward3 = run_parallel(reward_ins, tours_batch, adj=vars['adj'], service=vars['service_time'], clss=vars['clss'], k = 3)
    return np.float32([reward1, reward2, reward3]).T

def get_Ts_RL(vars, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
        
    if not isinstance(vars, dict):
        vars = convert_vars_np(vars)
                                        
    reward1 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 1)
    reward2 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 2)
    reward3 = run_parallel(reward_ins, tours_batch, vars['adj'], vars['service_time'], vars['clss'], k = 3)
    return np.float32([reward1, reward2, reward3]).T

@nb.njit(nb.float32(nb.int32[:, :], nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32), nogil=True)
def reward_ins(tours, adj, service, clss, k):
    r = 0.0
    for tour in tours:
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 0:
            continue
        candidate = tour[:pos[-1] + 1]
        length = calc_length(adj, service, candidate)
        r = max(r, length)
    return r 

@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32[:], nb.int32), nogil=True)
def reward_in(adj, service, clss, tour, k):
    r = 0.0
    pos = np.where(clss[tour] == k)[0]
    if len(pos) > 0:
        candidate = tour[:pos[-1] + 1]
        length = calc_length(adj, service, candidate)
        r = max(r, length)
    return r