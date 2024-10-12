import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numba as nb
import numpy as np
from common.nb_utils import calc_length, calc_demand
from common.cal_reward import reward_in
from common.consts import *

@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.float32[:], nb.float32[:], nb.int32[:], nb.int32[:]), nogil=True)
def once_interP(adj, service, demand, remain_demand, sub1, sub2):
    start, end, min_delta = 0, 0, 0

    best = max(calc_length(adj, service, sub1[1:]) + adj[sub1[0], sub1[1]],
               calc_length(adj, service, sub2[1:]) + adj[sub2[0], sub2[1]])
    
    demand_best = calc_demand(demand, sub1[1:]), calc_demand(demand, sub2[1:])

    for i in range(1, len(sub1)):
        for j in range(1, len(sub2)):
            candidate1 = sub1.copy()
            candidate2 = sub2.copy()
            candidate1[i], candidate2[j] = candidate2[j], candidate1[i]
            candidate_demand = calc_demand(demand, candidate1[1:]), calc_demand(demand, candidate2[1:])
            exceed_demand = (candidate_demand[0] - demand_best[0] > remain_demand[0]) or \
                            (candidate_demand[1] - demand_best[1] > remain_demand[1])
            if exceed_demand:
                continue

            length = max(calc_length(adj, service, candidate1[1:]) + adj[candidate1[0], candidate1[1]],
                         calc_length(adj, service, candidate2[1:]) + adj[candidate2[0], candidate2[1]])

            change = length - best
            if change < min_delta:
                start, end, min_delta, best, demand_best = i, j, change, length, candidate_demand

    if min_delta < -1e-6:
        sub1[start], sub2[end] = sub2[end], sub1[start]
        return min_delta
    else:
        return 0.0


@nb.njit(nb.float32(nb.float32[:, :], nb.float32[:], nb.int32[:], nb.float32[:], nb.float32[:], nb.int32[:], nb.int32[:], nb.int32), nogil=True)
def once_interU(adj, service, clss, demand, remain_demand, sub1, sub2, k):
    start, end, min_delta = 0, 0, 0

    best = np.zeros(k)
    for t in range(1, k+1):
        best[t-1] = max(reward_in(adj, service, clss, sub1, k=t), 
                        reward_in(adj, service, clss, sub2, k=t))
    length = np.zeros(k)
    demand_best = calc_demand(demand, sub1[1:]), calc_demand(demand, sub2[1:])
    for i in range(1, len(sub1)):
        for j in range(1, len(sub2)):
            candidate1 = sub1.copy()
            candidate2 = sub2.copy()
            candidate1[i], candidate2[j] = candidate2[j], candidate1[i]
            candidate_demand = calc_demand(demand, candidate1[1:]), calc_demand(demand, candidate2[1:])
            exceed_demand = (candidate_demand[0] - demand_best[0] > remain_demand[0]) or \
                            (candidate_demand[1] - demand_best[1] > remain_demand[1])

            if exceed_demand:
                continue
            
            for t in range(1, k+1):
                length[t-1] = max(reward_in(adj, service, clss, candidate1, k = t), 
                                  reward_in(adj, service, clss, candidate2, k = t))
            change = 0
            for t in range(k):
                c = length[t] - best[t]
                if c > 0:
                    break
                change += c*(10**(k-t))

            if change < min_delta:
                start, end, min_delta, best, demand_best = i, j, change, length, candidate_demand

    if min_delta < -1e-6:
        sub1[start], sub2[end] = sub2[end], sub1[start]
        return min_delta
    else:
        return 0.0

@nb.njit(nb.int32[:,:](nb.int32[:, :], nb.float32[:, :], nb.float32[:], nb.int32[:], nb.float32[:], nb.int32), nogil=True)
def interP(tours, adj, service, clss, demand, k):
    change = True
    it = 0
    remain_demand = np.ones(2, np.float32)
    while change and it < EPS:
        change = False
        for i in range(len(tours) - 1):
            for j in range(i + 1, len(tours)):
                pos1 = np.where(clss[tours[i]] == k)[0]
                pos2 = np.where(clss[tours[j]] == k)[0]
                if len(pos1) <= 0 or len(pos2) <= 0:
                    continue

                sub1 = tours[i][pos1[0] - 1: pos1[-1] + 1]
                sub2 = tours[j][pos2[0] - 1: pos2[-1] + 1]

                sub_change = -1.0
                sub_it = 0
                while sub_change < -1e-6 and sub_it < EPS:
                    remain_demand[0] = 1 - demand[tours[i]].sum()
                    remain_demand[1] = 1 - demand[tours[j]].sum()
                    sub_change = once_interP(adj, service, demand, remain_demand, sub1, sub2)
                    sub_it += 1
                    if sub_it >= 2:
                        change = True
        it += 1
    return tours

@nb.njit(nb.int32[:,:](nb.int32[:, :], nb.float32[:, :], nb.float32[:], nb.int32[:], nb.float32[:], nb.int32), nogil=True)
def interU(tours, adj, service, clss, demand, k):
    change = True
    it = 0
    remain_demand = np.ones(2, np.float32)
    while change and it < EPS:
        change = False
        for i in range(len(tours) - 1):
            for j in range(i + 1, len(tours)):
                pos1 = np.where(clss[tours[i]] == k)[0]
                pos2 = np.where(clss[tours[j]] == k)[0]
                if len(pos1) <= 0 or len(pos2) <= 0:
                    continue
                
                sub1 = tours[i][: pos1[-1] + 1]
                sub2 = tours[j][: pos2[-1] + 1]

                sub_change = -1.0
                sub_it = 0
                while sub_change < -1e-6 and sub_it < EPS:
                    remain_demand[0] = 1 - demand[tours[i]].sum()
                    remain_demand[1] = 1 - demand[tours[j]].sum()
                    sub_change = once_interU(adj, service, clss, demand, remain_demand, sub1, sub2, k)
                    sub_it += 1
                    if sub_it >= 2:
                        change = True
        it += 1
    return tours