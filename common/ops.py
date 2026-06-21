import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

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


def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def _batchify_single(x, repeats):
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])

def batchify(x, shape):
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def get_log_likelihood(logprobs, actions=None, mask=None, return_sum: bool = True):
    """Get log likelihood of selected actions.
    Note that mask is a boolean tensor where True means the value should be kept.

    Args:
        logprobs: Log probabilities of actions from the model (batch_size, seq_len, action_dim).
        actions: Selected actions (batch_size, seq_len).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        return_sum: Whether to return the sum of log probabilities or not. Defaults to True.
    """
    # Optional: select logp when logp.shape = (bs, dec_steps, N)
    if actions is not None and logprobs.dim() == 3:
        logprobs = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        logprobs[~mask] = 0

    # Calculate log_likelihood
    if return_sum:
        return logprobs.sum(1)  # [batch]
    else:
        return logprobs  # [batch, decode_len]

def _unbatchify_single(x, repeats: int):
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(x, shape):
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x

def unbatchify_and_gather(x, idx, n):
    """first unbatchify a tensor by n and then gather (usually along the unbatchified dimension)
    by the specified index
    """
    x = unbatchify(x, n)
    return gather_by_index(x, idx, dim=idx.dim())

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
    # print("---> prob_idxs", idx.shape)
    return idx
        
def refine_routes(actions, demands, max_vehicles=3, **kwargs):
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

    # print("---> refine_routes", ret.shape)
    return ret

def convert_prob(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)


def convert_adjacency_matrix(n1, n2, d):
    n1, n2 = n1.astype(int), n2.astype(int)
    n = len(np.unique([n1, n2]))
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0)
    adj[n1, n2] = d
    return adj

def floyd_warshall(adj):
    dms = adj.copy()
    for k in range(adj.shape[0]):
        dms = np.minimum(dms, dms[:, k, None] + dms[None, k, :])
    return dms

def dist_edges(dms, n1, n2):
    dms = dms.copy()
    n1 = n1.astype(int)
    n2 = n2.astype(int)
    go_from = np.hstack([[0], n2])[..., None]
    go_to = np.hstack([[0], n1])[None, ...]
    dms = dms[go_from, go_to]
    return dms.astype(np.float32)

def dist_edges_from_file(es):
    if isinstance(es, str):
        es = np.load(es)
    es_cat = np.concatenate([es['req'], es['nonreq']], axis=0)
    adj = convert_adjacency_matrix(es_cat[:, 0], es_cat[:, 1], es_cat[:, -1])
    dms = floyd_warshall(adj)
    dms = dist_edges(dms, es['req'][:, 0], es['req'][:, 1])
    return dms

def import_instance(es, M=None):
    # M (fleet size) is a SOLVE-time parameter, not an instance property: the
    # instance (arcs, demands, C) is identical for any M because C = sum(q)/3 +
    # 0.5 has a fixed /3 (paper F5). Pass M to override the nominal value stored
    # in the .npz; leave it None to use the stored default.
    if isinstance(es, str):
        es = np.load(es)
    C = es['C']
    P = [i for i in range(1, es['P']+1)]
    n_vehicle = int(es['M']) if M is None else int(M)
    M = [i for i in range(n_vehicle)]
    dms = dist_edges_from_file(es)
    
    es_req = es['req']
    es_req = np.vstack([[0]*6, es_req])
    edge_indxs = np.int32(es_req[:, :2])
    demands = np.float32(es_req[:, 2]) / C
    clss = np.int32(es_req[:, 3])
    s = np.float32(es_req[:, 4])
    d = np.float32(es_req[:, 5])
    return dms, P, M, demands, clss, s, d, edge_indxs

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
