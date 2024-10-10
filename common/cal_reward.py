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
    bs = len(tours_batch)     
    reward1 = run_parallel(reward_ins, [1]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    # reward2 = run_parallel(reward_ins, [2]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    # reward3 = run_parallel(reward_ins, [3]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    return reward1

def get_Ts(vars, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
        
    if not isinstance(vars, dict):
        vars = convert_vars_np(vars)
    bs = len(tours_batch)                                   
    reward1 = run_parallel(reward_ins, [1]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    reward2 = run_parallel(reward_ins, [2]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    reward3 = run_parallel(reward_ins, [3]*bs, vars['adj'], vars['service_time'], vars['clss'], tours_batch)
    return np.float32([reward1, reward2, reward3]).T

@nb.njit(nb.float32(nb.int32, nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32[:, :]), nogil=True)
def reward_ins(k, adj, service, clss, tours):
    r = 0.0
    for tour in tours:
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 0:
            continue
        candidate = tour[:pos[-1] + 1]
        length = calc_length(adj, service, candidate)
        r = max(r, length)
    return r + 1000*max(0, len(tours) - 2)

@nb.njit(nb.float32(nb.int32, nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32[:]), nogil=True)
def reward_in(k, adj, service, clss, tour):
    r = 0.0
    pos = np.where(clss[tour] == k)[0]
    if len(pos) > 0:
        candidate = tour[:pos[-1] + 1]
        length = calc_length(adj, service, candidate)
        r = max(r, length)
    return r
