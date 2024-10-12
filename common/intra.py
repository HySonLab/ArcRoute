import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
from common.cal_reward import reward_in
from common.nb_utils import calc_length
from common.consts import *

@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.int32[:]), nogil=True)
def once_intraP(adj, service, sub):
    n = len(sub)
    best = calc_length(adj, service, sub[1:]) + adj[sub[0], sub[1]]
    start, end, min_delta = 0, 0, 0

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            candidate = sub.copy()
            candidate[i:j+1] = np.flip(candidate[i:j+1])
            length = calc_length(adj, service, candidate[1:]) + adj[candidate[0], candidate[1]]
            change = length - best
            if change < min_delta:
                start, end, min_delta, best = i, j, change, length

    if min_delta < -1e-6:
        sub[start:end+1] = np.flip(sub[start:end+1])
        return min_delta
    else:
        return 0.0

@nb.njit(nb.int32[:,:](nb.int32[:, :], nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32), nogil=True)
def intraP(tours, adj, service, clss, k):
    for tour_idx in range(len(tours)):
        tour = tours[tour_idx]
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 1:
            continue

        it = 0
        change = -1.0
        sub = tour[pos[0] - 1: pos[-1] + 1]
        while change < -1e-6 and it < EPS:
            change = once_intraP(adj, service, sub)
            it += 1
    return tours


@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32[:], nb.int32), nogil=True)
def once_intraU(adj, service, clss, sub, k):
    n = len(sub)
    best = calc_length(adj, service, sub[1:]) + adj[sub[0], sub[1]]

    best = np.zeros(k)
    for t in range(1, k+1):
        best[t-1] = reward_in(adj, service, clss, sub, k = t)
        
    start, end, min_delta = 0, 0, 0
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            candidate = sub.copy()
            candidate[i:j+1] = np.flip(candidate[i:j+1])
            length = np.zeros(k)
            for t in range(1, k+1):
                length[t-1] = reward_in(adj, service, clss, candidate, k = t)
            change = 0
            for t in range(k):
                c = length[t] - best[t]
                if c > 0:
                    break
                change += c*(10**(k-t))
            if change < min_delta:
                start, end, min_delta, best = i, j, change, length

    if min_delta < -1e-6:
        sub[start:end+1] = np.flip(sub[start:end+1])
        return min_delta
    else:
        return 0.0

@nb.njit(nb.int32[:,:](nb.int32[:, :], nb.float32[:, :], nb.float32[:], nb.int32[:], nb.int32), nogil=True)
def intraU(tours, adj, service, clss, k):
    for tour_idx in range(len(tours)):
        tour = tours[tour_idx]
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 1:
            continue

        it = 0
        change = -1.0
        sub = tour[: pos[-1] + 1]
        while change < -1e-6 and it < EPS:
            change = once_intraU(adj, service, clss, sub, k)
            it += 1
    return tours