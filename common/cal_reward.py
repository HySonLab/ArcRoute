import numpy as np
import torch
from common.local_search import lsRL

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

def calc_reward(action, td, pos_val=[1,2,3], **kwargs):
    tours = action_to_tours(action)
    prior = td['clss'][tours]
    total_time = td['service_times'][tours]
    shortest_traversal_time = td['adj'][tours[:, :-1], tours[:, 1:]]
    total_time[:, 1:] += shortest_traversal_time
    total_time = torch.cumsum(total_time, dim=1)
    # if kwargs.get("local_search", False):
    #     tours = lsRL(td, tours)
    rs = []
    for p in pos_val:
        pos = torch.nonzero(prior == p, as_tuple=True)
        if len(pos[0]) == 0:
            pos = [[0], [0]]
        rs.append(total_time[pos].max())
    if kwargs.get("return_list", False):
        return rs
    if kwargs.get("return_numpy", False):
        return np.array(rs)
    return torch.tensor(rs)