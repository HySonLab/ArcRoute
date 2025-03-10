import numpy as np

def calc_length(adj, service, sub):
    s = service[sub].clone()
    a = adj[sub[:-1], sub[1:]].clone()
    s[1:] += a
    return s.sum()

def once_intraU(adj, service, sub):
    n = len(sub)
    best = calc_length(adj, service, sub)
    start, end, min_delta = 0, 0, 0

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            candidate = sub.copy()
            candidate[i:j+1] = np.flip(candidate[i:j+1])
            length = calc_length(adj, service, candidate)
            change = length - best
            if change < min_delta:
                start, end, min_delta, best = i, j, change, length

    if min_delta < -1e-6:
        sub[start:end+1] = np.flip(sub[start:end+1])
        return min_delta, sub
    else:
        return 0.0, sub

def intraU(adj, service, prior, tours, pos_val=[1,2,3]):
    prior = prior.numpy()
    for tour_idx in range(len(tours)):
        tour = tours[tour_idx]
        it = 0
        change = -1.0
        pos_ids = dict.fromkeys(pos_val, 0)
        pos = len(tour) - 1
        count = 0
        _prior = prior[tour]
        while count < len(pos_val) and pos > 0:
            if _prior[pos] in pos_val and pos_ids[_prior[pos]] == 0:
                pos_ids[_prior[pos]] = pos
                count += 1
            pos -= 1

        for k in pos_val:
            if pos_ids[k] == 0:
                continue
            pos = pos_ids[k]
            if pos <= 1:
                continue
            tour = tours[tour_idx, :pos]
            while change < -1e-6 and it < 100:
                change, tour = once_intraU(adj, service, tour)
                it += 1
            tours[tour_idx, :pos] = tour
    return tours

def lsRL(td, tours):
    tours = intraU(td['adj'], td['service_times'], td['clss'], tours)
    return tours

