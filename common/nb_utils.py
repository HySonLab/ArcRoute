import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
from common.ops import run_parallel

@nb.njit(nb.float32[:](nb.float32[:, :], nb.int32[:], nb.int32[:]), nogil=True)
def index2d(x, a1, a2):
    return np.float32([x[i, j] for i, j in zip(a1, a2)])

@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.int32[:]), nogil=True)
def calc_length(adj, service_time, sub):
    length = np.sum(service_time[sub]) + np.sum(index2d(adj, sub[:-1], sub[1:]))
    return length

@nb.njit(nb.float32(nb.float32[:], nb.int32[:]), nogil=True)
def calc_demand(demands, sub):
    return np.sum(demands[sub])

@nb.njit(nb.int32[:,:](nb.int32[:]), nogil=True)
def gen_tours(action):
    idxs = [0] + [i+1 for i in range(len(action)) if action[i] == 0] + [len(action)]
    tours = []
    maxlen = 0
    for i,j in zip(idxs[:-1], idxs[1:]):
        a = action[i:j]
        if a.sum() == 0:
            continue
        tours.append(a)
        maxlen = max(maxlen, len(a))
    padded = np.zeros((len(tours), maxlen+2), dtype=np.int32)
    for idx, tour in enumerate(tours):
        padded[idx][1:len(tour)+1] = tour
    return padded

def gen_tours_batch(actions):
    if isinstance(actions, list):
        actions = np.int32(actions)
    if not isinstance(actions, np.ndarray):
        actions = actions.cpu().numpy().astype(np.int32)
       
    tours_batch = run_parallel(gen_tours, actions)
    return tours_batch

@nb.njit(nb.int32[:](nb.int32[:,:], nb.int32), nogil=True)
def deserialize_tours(tours, n):
    new_action = []
    for tour in tours:
        j = len(tour) - 1
        while tour[j] == 0 and j >= 0: j -= 1
        new_action.extend(tour[1:j+2])
    while(len(new_action) < n): new_action.append(0)
    while(len(new_action) > n): new_action.pop(-1)
    return np.int32(new_action)

def deserialize_tours_batch(tours_batch, n):
    new_actions = run_parallel(deserialize_tours, tours_batch, n=n)
    return np.array(new_actions)