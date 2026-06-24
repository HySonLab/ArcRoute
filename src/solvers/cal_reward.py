import numpy as np
import torch

def action_to_tours(action):
    zero_indices = np.where(action == 0)[0]
    split_indices = np.concatenate(([-1], zero_indices, [len(action)]))
    lengths = np.diff(split_indices) - 1
    valid_lengths = lengths[lengths > 0]
    nonzero_action = action[action != 0]
    tours = np.split(nonzero_action, np.cumsum(valid_lengths)[:-1])
    # Tìm độ dài lớn nhất để padding
    max_len = max(len(r) for r in tours)

    # Padding để đảm bảo tất cả có cùng kích thước
    padded = np.zeros((len(tours), max_len+2), dtype=np.int32)
    for idx, tour in enumerate(tours):
        padded[idx][1:len(tour)+1] = tour
    return padded

def get_Ts(vars, actions, pos_val=[1, 2, 3]):
    """Hierarchical completion times for a batch of actions.

    vars : {'adj': dms, 'service_time': s, 'clss': clss, 'demand': demands, 'nv': M}
    actions : 2D array (batch, n) — flat arc-index sequences with 0-separators.
    Returns : np.ndarray shape (batch, len(pos_val)).
    """
    from solvers.scheduler import Scheduler
    td = {
        'adj':           vars['adj'],
        'service_times': vars['service_time'],
        'clss':          vars['clss'],
        'demand':        vars['demand'],
    }
    M = vars.get('nv', None)
    sched = Scheduler(pos_val=tuple(pos_val))
    results = []
    for action in actions:
        _, T = sched(np.asarray(action), td, M=M)
        results.append(T)
    return np.array(results, dtype=np.float64)


def calc_reward(action, td, pos_val=[1,2,3], **kwargs):
    """Hierarchical completion times (T_1,...,T_p) for one instance.

    Delegates to the Scheduler Φ (dynamic_plan Phase 1): the M-agnostic policy
    provides the arc ORDER `action`, the Scheduler partitions it into the routes
    of M vehicles (multi-trip when needed) and computes the T_k. M is read from
    td['num_vehicle']; falls back to k_min (capacity-minimum, parallel) if absent.
    """
    from solvers.scheduler import Scheduler  # local import to avoid cycles
    sched = Scheduler(variant=kwargs.get("variant", "P"), pos_val=tuple(pos_val))
    M = kwargs.get("M", None)
    if M is None and "num_vehicle" in td:
        M = int(td["num_vehicle"])
    _, T = sched(action, td, M=M)  # M=None -> Scheduler uses k_min (parallel)
    rs = [float(x) for x in T]
    # if kwargs.get("local_search", False):
    #     tours = lsRL(td, tours)
    if kwargs.get("return_list", False):
        return rs
    if kwargs.get("return_numpy", False):
        return np.array(rs)
    return torch.tensor(rs)