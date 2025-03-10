import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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

def calc_reward(action, td, pos_val=[1,2,3], **kwargs):
    tours = action_to_tours(action)
    prior = td['clss'][tours]
    total_time = td['service_times'][tours]
    shortest_traversal_time = td['adj'][tours[:, :-1], tours[:, 1:]]
    total_time[:, 1:] += shortest_traversal_time
    total_time = torch.cumsum(total_time, dim=1)
    if kwargs.get("local_search", False):
        tours = intraU(td['adj'], td['service_times'], td['clss'], tours)
    rs = []
    for p in pos_val:
        pos = torch.nonzero(prior == p, as_tuple=True)
        if len(pos[0]) == 0:
            pos = [[0], [0]]
        rs.append(total_time[pos].max())
    return torch.tensor(rs)

def run_parallel(operation, *param_set, **kwargs):
    def padding(batch):
        if kwargs.get("return_numpy", False):
            max_len = max([len(x) for x in batch])
            padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in batch])
        else:
            padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return padded

    class WrapDataset(Dataset):
        def __init__(self, operation, *param_set, **kwargs):
            self.operation = operation
            self.param_set = param_set
            self.kwargs = kwargs

        def __len__(self):
            return len(self.param_set[0])

        def __getitem__(self, i):
            sliced_params = [param[i] for param in self.param_set]
            return self.operation(*sliced_params, **self.kwargs)
        
        @classmethod
        def collate_fn(self, batch):
            return padding(batch)

        
    dataloader = DataLoader(
        WrapDataset(operation, *param_set, **kwargs),
        batch_size=len(param_set[0])//kwargs.get("num_epochs", 24),
        shuffle=False,
        num_workers=kwargs.get("num_workers", 10),
        collate_fn=WrapDataset.collate_fn if kwargs.get("must_padding", False) else None
    )
    
    is_tqdm=kwargs.get("is_tqdm", False)
    ts = []
    for t in tqdm(dataloader) if is_tqdm else dataloader:
        ts.append(t)
    
    # if kwargs.get("ts_padding", False):
    #     padded = padding(ts)
    #     return padded
    if kwargs.get("return_list", False):
        return ts
    if kwargs.get("return_numpy", False):
        return np.concatenate(ts, 0)
    return torch.cat(ts, 0)


def find_best_params(operation, *param_set, **kwargs):
    worker_options = [1, 2, 4, 8, 16, 24]
    epoch_options = [10, 20, 50, 100, 200]

    best_params = {'num_workers': 1, 'num_epochs': 10, 'elapsed': float('inf')}
    for num_workers in worker_options:
        for num_epochs in epoch_options:
            
            start = time.time()
            kparams = {'num_workers': num_workers, 'num_epochs': num_epochs, **kwargs}
            _ = run_parallel(operation, *param_set, **kparams)
            elapsed = time.time() - start

            print({'num_workers': num_workers, 'num_epochs': num_epochs, 'elapsed': elapsed})
            if elapsed < best_params['elapsed']:
                best_params = {'num_workers': num_workers, 'num_epochs': num_epochs, 'elapsed': elapsed}

    return best_params

def refine_routes(actions, demands, max_vehicles=5, **kwargs):
    actions = actions.tolist()
    # Khởi tạo danh sách các route và sức chứa
    routes = [[] for _ in range(max_vehicles)]  # Danh sách các route cho mỗi vehicle
    capacities = [0.0] * max_vehicles  # Tổng demand của mỗi vehicle
    
    # Chỉ số vehicle hiện tại
    vehicle_idx = 0
    
    for arc in actions:
        if arc == 0:
            # Chuyển sang vehicle tiếp theo nếu còn chỗ
            if vehicle_idx + 1 < max_vehicles:
                vehicle_idx += 1
            continue
        
        demand = demands[arc]
        
        # Kiểm tra xem vehicle hiện tại có thể chứa arc không
        if capacities[vehicle_idx] + demand <= 1.0:
            routes[vehicle_idx].append(arc)
            capacities[vehicle_idx] += demand
        else:
            # Thử đặt arc vào các vehicle tiếp theo
            placed = False
            for i in range(vehicle_idx + 1, max_vehicles):
                if capacities[i] + demand <= 1.0:
                    routes[i].append(arc)
                    capacities[i] += demand
                    placed = True
                    break
            # Nếu không tìm được vehicle nào, đặt vào vehicle hiện tại (có thể vượt quá sức chứa)
            if not placed and vehicle_idx == max_vehicles - 1:
                routes[vehicle_idx].append(arc)
                capacities[vehicle_idx] += demand
    ret = []
    for route in routes:
        if len(route) > 0:
            ret.extend([0] + route)

    ret += [0]*(max_vehicles + len(demands) - len(ret))
    if kwargs.get("return_numpy", False):
        ret = np.array(ret[1:], dtype=np.int64)
    else:
        ret = torch.tensor(ret[1:])

    print("---> refine_routes", ret.shape)
    return ret

def prob_idxs(a1, a2, **kwargs):
    idx = []
    i,j = 0,0
    while j < len(a2) and i < len(a1):
        if a2[j] == a1[i]:
            idx.append(i)
            j += 1
        i += 1
    while len(idx) < kwargs['npad']:
        idx.append(0)
    if kwargs.get("return_numpy", False):
        idx = np.array(idx, dtype=np.int64)
    else:
        idx = torch.tensor(idx, dtype=torch.int64)
    print("---> prob_idxs", idx.shape)
    return idx

if __name__ == "__main__":
    td = torch.load('../data/td.pt', weights_only=False).cpu()
    # tours_batch = torch.load('tours_batch.pt', weights_only=False)
    actions = torch.load('../data/actions.pt', weights_only=False)
    rewards = torch.load('../data/rewards.pt', weights_only=False)
    actions = torch.cat([actions]*10, 0).cpu().numpy()
    td = torch.cat([td]*10, 0)
    # t1 = time.time()
    # r1 = run_parallel(calc_reward, actions[:3], td[:3], num_workers=3, num_epochs=3)
    # print(r1)
    # print(time.time() - t1)
    # param_set = (actions, td)
    # best_params = find_best_params(calc_reward, *param_set, is_tqdm=False)
    # print("Optimal Parameters:", best_params)

    # t1 = time.time()
    # r1 = run_parallel(calc_reward, *param_set, **best_params)
    # print("Optimal Time:", time.time() - t1)

    # t1 = time.time()
    # ts = []
    # for i in tqdm(range(len(actions))):
    #     t = calc_reward(actions[i], td[i])
    #     ts.append(t[None])
    # r2 = torch.cat(ts, 0)
    # print(time.time() - t1)

    # print((r1 - r2).mean())
    # print(calc_reward(actions[0], td[0]))

    # routes = torch.tensor([17, 19, 1, 0, 14, 13, 18, 2, 9, 16, 15, 20, 7, 10, 5, 3, 8, 6, 4, 12, 11, 0, 2, 3, 5])
    # action = refine_routes(routes, td[0]['demand'], max_vehicles=5)
    # print(action)
    
    # print("--- action")
    # print(actions[0])
    # print("--- demand")
    # print(td[0]['demand'])
    # action = refine_routes(actions[0], td[0]['demand'], max_vehicles=5)
    # print(action)
    # print(td[0]['demand'][action][:3].sum())
    # print(td[0]['demand'][action][4:].sum())
    # ts = []
    # for i in tqdm(range(len(actions))):
    #     t = refine_routes(actions[i], td[i]['demand'], max_vehicles=5)
    #     ts.append(t[None])


    # print(refine_routes(actions[0], td[0]['demand'], max_vehicles=5))
    # t1 = time.time()
    # r1 = run_parallel(refine_routes, actions, td['demand'], max_vehicles=5, num_workers=24, num_epochs=100)
    # print(r1)
    # print("Parallel Time:", time.time() - t1)

    # param_set = (actions, td['demand'])
    # best_params = find_best_params(refine_routes, *param_set, max_vehicles=5, is_tqdm=True, must_padding=True)
    # print("Optimal Parameters:", best_params)
    a1 = actions
    a2 = run_parallel(refine_routes, actions, td['demand'], max_vehicles=3, num_workers=24, num_epochs=100)
    # # print(a1)
    # # print(a2)
    # # print(a1)
    # # print(a2)
    # # print(prob_idxs(a1, a2, return_numpy=True))
    r = run_parallel(prob_idxs, a1, a2, num_workers=24, num_epochs=100, must_padding=True, is_tqdm=True, npad=23)
    print(r.shape)
    # print(r[0])
    # print(r[1])
    # print(r[2])