import numpy as np
import numba as nb
from common.ops import  setup_vars, run_parallel, gen_tours_batch
def local_search(vars=None, td=None, actions=None, tours_batch=None, epochs=100):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
        
    if td is not None:
        adjs, ss, clss = setup_vars(td)
    else:
        adjs, ss, clss = vars

    bs = len(tours_batch)
    for k in [1,2,3]:
        for _ in range(3):
            tours_batch = run_parallel(intra, [k]*bs, ss, adjs, clss, tours_batch, epochs=epochs)
            tours_batch = run_parallel(inner, [k]*bs, ss, adjs, clss, tours_batch, epochs=epochs)
    return tours_batch

@nb.njit(nb.float32(nb.float32[:], nb.float32[:,:], nb.int32[:]), nogil=True)
def once_intra(service_time, adj, sub):
    n = len(sub)
    p = q = 0
    delta = 0
    
    best = sum([service_time[i] for i in sub[1:]]) + sum([adj[i,j] for i, j in zip(sub[:-1],sub[1:])])
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            candidate = sub.copy()
            candidate[i:j+1] = np.flip(candidate[i:j+1])
            length = sum([service_time[i] for i in candidate[1:]]) + sum([adj[i,j] for i, j in zip(candidate[:-1],candidate[1:])])
            change = length - best
            if change < delta:
                p, q, delta, best = i, j, change, length
                        
    if delta < -1e-6:
        sub[p:q+1] = np.flip(sub[p:q+1])
        return delta
    else:
        return 0.0
    
@nb.njit(nb.int32[:,:](nb.int32, nb.float32[:], nb.float32[:, :], nb.int32[:], nb.int32[:, :], nb.int64), nogil=True)
def intra(k, service_time, adj, clss, tours, epochs=1000):
    for tour in tours:
        pos = np.where(clss[tour] == k)[0]
        if len(pos) <= 0:
            continue
        if len(pos) >= 2:
            iterations = 0
            min_change = -1.0
            sub = tour[pos[0] - 1: pos[-1] + 1]
            while min_change < -1e-6 and iterations < epochs:
                min_change = once_intra(service_time, adj, sub)
                iterations += 1
    return tours

@nb.njit(nb.float32(nb.float32[:], nb.float32[:,:], nb.int32[:], nb.int32[:]), nogil=True)
def once_inner(service_time, adj, sub1, sub2):
    p = q = 0
    delta = 0
    
    best = max(sum([service_time[i] for i in sub1[1:]]) + sum([adj[i,j] for i, j in zip(sub1[:-1],sub1[1:])]),
    sum([service_time[i] for i in sub2[1:]]) + sum([adj[i,j] for i, j in zip(sub2[:-1],sub2[1:])]))

    for i in range(1, len(sub1)):
        for j in range(1, len(sub2)):
            candidate1 = sub1.copy()
            candidate2 = sub2.copy()
            candidate1[i], candidate2[j] = candidate2[j], candidate1[i]
            length = max(sum([service_time[i] for i in candidate1[1:]]) + sum([adj[i,j] for i, j in zip(candidate1[:-1],candidate1[1:])]),
            sum([service_time[i] for i in candidate2[1:]]) + sum([adj[i,j] for i, j in zip(candidate2[:-1],candidate2[1:])]))
            change = length - best
            if change < delta:
                p, q, delta, best = i, j, change, length
                        
    if delta < -1e-6:
        sub1[p], sub2[q] = sub2[q], sub1[p]
        return delta
    else:
        return 0.0

@nb.njit(nb.int32[:,:](nb.int32, nb.float32[:], nb.float32[:, :], nb.int32[:], nb.int32[:, :], nb.int64), nogil=True)
def inner(k, service_time, adj, clss, tours, epochs=1000):
    change = False
    iterations = 0
    while not change and iterations < epochs:
        for i in range(len(tours)-1):
            for j in range(i+1, len(tours)):
                pos1 = np.where(clss[tours[i]] == k)[0]
                pos2 = np.where(clss[tours[j]] == k)[0]
                if len(pos1) <= 0 or len(pos2) <= 0:
                    continue
                else:
                    sub1 = tours[i][pos1[0] - 1: pos1[-1] + 1]
                    sub2 = tours[j][pos2[0] - 1: pos2[-1] + 1]

                    _min_change = -1.0
                    _iterations=0
                    while _min_change < -1e-6 and _iterations < epochs:
                        _min_change = once_inner(service_time, adj, sub1, sub2)
                        _iterations+=1
                        change = change | (_iterations < 2)
        iterations += 1
    return tours