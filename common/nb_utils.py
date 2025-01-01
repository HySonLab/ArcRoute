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

@nb.njit(nb.int32[:](nb.int32[:], nb.int32[:]), nogil=True)
def prob_idxs(a1, a2):
    idx = []
    i,j = 0,0
    while j < len(a2) and i < len(a1):
        if a2[j] == a1[i]:
            idx.append(i)
            j += 1
        i += 1
    return np.array(idx, dtype=np.int32)
        
@nb.njit(nb.int32[:](nb.int32[:], nb.float32[:], nb.int32), nogil=True)
def refine_routes(actions, demands, max_vehicles=5):
    # Initialize routes
    routes = np.zeros((max_vehicles, len(actions)), dtype=np.int32)
    capacities = np.zeros(max_vehicles, dtype=np.float32)
    route_lengths = np.zeros(max_vehicles, dtype=np.int32)

    # Current vehicle index
    vehicle_idx = 0
    for arc in actions:
        if arc == 0:
            # Move to the next vehicle if possible
            if vehicle_idx + 1 < max_vehicles:
                vehicle_idx += 1
            continue
        
        demand = demands[arc]

        # Check if the current vehicle can take this arc
        if capacities[vehicle_idx] + demand <= 1.0:
            routes[vehicle_idx, route_lengths[vehicle_idx]] = arc
            route_lengths[vehicle_idx] += 1
            capacities[vehicle_idx] += demand
        else:
            # Try to fit arc in any subsequent vehicle
            placed = False
            for i in range(vehicle_idx + 1, max_vehicles):
                if capacities[i] + demand <= 1.0:
                    routes[i, route_lengths[i]] = arc
                    route_lengths[i] += 1
                    capacities[i] += demand
                    placed = True
                    break
            # If no vehicle can take it, put it in the current vehicle (may exceed capacity if no option)
            if not placed:
                if vehicle_idx == max_vehicles - 1:
                    routes[vehicle_idx, route_lengths[vehicle_idx]] = arc
                    route_lengths[vehicle_idx] += 1
                    capacities[vehicle_idx] += demand

    # Flatten the routes while removing empty slots
    result = []
    for i in range(max_vehicles):
        if route_lengths[i] > 0:
            result.append(0)  # Start of a new route
            result.extend(routes[i, :route_lengths[i]])
    
    return np.array(result[1:], dtype=np.int32)  # Remove the leading 0

def convert_prob(x):
    # a = a - np.min(a) + np.float32(1e-4)
    # a = a / np.sum(a, axis=0)
    # return a
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)