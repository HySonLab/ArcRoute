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
    # C is generated as Σq/M_nominal + 0.5 (M-dependent). Pass M to override
    # the nominal fleet stored in the .npz at eval time; None uses the stored default.
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

def run_parallel2(func, items, **kwargs):
    """Map func(item, **kwargs) over items, return list of results."""
    return [func(item, **kwargs) for item in items]


def gen_tours(actions):
    """Flat action array [a1, a2, 0, a3, ...] -> list of routes.

    Each returned route is a numpy int32 array with sentinel depot 0s at the
    start and end, e.g. [0, a_i, ..., 0]. Empty segments (consecutive 0s) are
    skipped.
    """
    routes = []
    current = [0]
    for a in actions:
        if a == 0:
            if len(current) > 1:
                current.append(0)
                routes.append(np.array(current, dtype=np.int32))
                current = [0]
        else:
            current.append(int(a))
    if len(current) > 1:
        current.append(0)
        routes.append(np.array(current, dtype=np.int32))
    return routes


def deserialize_tours(tours, max_len):
    """List of routes (each [0, a_i, ..., 0]) -> flat actions array of length max_len.

    Sentinels are stripped from each route, routes are joined with single 0
    separators, and the result is zero-padded (or truncated) to ``max_len``.
    """
    flat = []
    for r in tours:
        r = np.asarray(r)
        flat.extend(r[1:-1].tolist())  # strip sentinels
        flat.append(0)                 # add separator
    # strip trailing separator
    flat = flat[:-1] if flat and flat[-1] == 0 else flat
    arr = np.zeros(max_len, dtype=np.int32)
    n = min(len(flat), max_len)
    arr[:n] = flat[:n]
    return arr


def deserialize_tours_batch(tours_list, nseq):
    """List of route-lists -> 2D flat action array, shape (batch, actual_max).

    Each input element is a list of routes (as produced by ``gen_tours``).
    """
    rows = [deserialize_tours(t, nseq + 10) for t in tours_list]
    actual_max = max((len(r) for r in rows), default=0)
    out = np.zeros((len(rows), actual_max), dtype=np.int32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r[:actual_max]
    return out


def convert_prob(x):
    """Softmax via log-sum-exp normalisation."""
    x = np.asarray(x, dtype=np.float64)
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)


def print_route_log(routes, req, demands, adj, Ts):
    """Print a formatted route log to stdout.

    Args:
        routes:  list of int32 arrays, each [0, arc, ..., 0].
        req:     raw required-arc array, shape (nseq, 6) — columns
                 [tail, head, demand, clss, service, traversal].
        demands: 1-D float array indexed by arc (0 = depot).
        adj:     arc-to-arc dead-heading cost matrix.
        Ts:      sequence of objective values (T1, T2, T3).
    """
    P = int(req[:, 3].max())
    arc_info = {
        i + 1: dict(
            tail=int(req[i, 0]),
            head=int(req[i, 1]),
            clss=int(req[i, 3]),
            service=float(req[i, 4]),
            demand=float(demands[i + 1]),
        )
        for i in range(len(req))
    }

    T_str = "  ".join(f"T{k + 1}={v:.4f}" for k, v in enumerate(Ts))
    print(f"\n{T_str}\n")

    for vid, route in enumerate(routes):
        arcs = route[1:-1]
        total_load = float(demands[arcs].sum()) if len(arcs) > 0 else 0.0
        bar = "=" * 60
        print(bar)
        print(f"  VEHICLE {vid + 1}   load={total_load:.4f}/1.0000")
        print(bar)

        chain = ["depot"]
        prev_head = None
        for arc_idx in arcs:
            info = arc_info[arc_idx]
            t, h = info["tail"], info["head"]
            if prev_head is not None and prev_head != t:
                chain.append(f"~~{prev_head}->{t}~~")
            chain.append(f"{t}->{h}")
            prev_head = h
        chain.append("depot")

        line, indent = "  ", "    "
        for k, part in enumerate(chain):
            sep = " -> " if k > 0 else ""
            if len(line) + len(sep) + len(part) > 70:
                print(line)
                line = indent + part
            else:
                line += sep + part
        print(line)
        print()

        hdr = (f"  {'seq':>3}  {'arc':>3}  {'edge':>7}  {'cls'}"
               f"  {'demand':>7}  {'service':>7}  {'dead':>7}  {'cum_load':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        cum = 0.0
        for pos, arc_idx in enumerate(arcs):
            info = arc_info[arc_idx]
            prev_arc = arcs[pos - 1] if pos > 0 else 0
            dead = float(adj[prev_arc, arc_idx])
            cum += info["demand"]
            print(
                f"  {pos + 1:>3}  {arc_idx:>3}  "
                f"{info['tail']:>2}->{info['head']:>2}  "
                f"  {info['clss']}  "
                f"{info['demand']:>7.4f}  "
                f"{info['service']:>7.4f}  "
                f"{dead:>7.4f}  "
                f"{cum:>8.4f}"
            )
        print()

        for p in range(1, P + 1):
            class_arcs = [(pos, a) for pos, a in enumerate(arcs) if arc_info[a]["clss"] == p]
            if not class_arcs:
                continue
            class_load = sum(arc_info[a]["demand"] for _, a in class_arcs)
            print(f"  Class {p}  ({len(class_arcs)} arcs, load={class_load:.4f})")
            for pos, arc_idx in class_arcs:
                info = arc_info[arc_idx]
                prev_arc = arcs[pos - 1] if pos > 0 else 0
                dead = float(adj[prev_arc, arc_idx])
                print(
                    f"    [seq {pos + 1:2d}] arc {arc_idx:2d}  "
                    f"{info['tail']:2d} -> {info['head']:2d}   "
                    f"demand={info['demand']:.4f}  "
                    f"service={info['service']:.4f}  "
                    f"dead={dead:.4f}"
                )
            print()
    print("=" * 60)


def check_feasibility(demands, nv):
    """Return True if fleet can cover total demand; print error and return False otherwise."""
    total = float(demands[1:].sum())
    if total > nv * 1.0 + 1e-6:
        print(f"ERROR: No feasible solution found with {nv} vehicle(s).")
        print(f"  Total demand = {total:.4f}  >  fleet capacity = {nv:.1f}")
        print(f"  Minimum vehicles required: {int(np.ceil(total - 1e-6))}")
        return False
    return True


def save_sol(sol_path, routes, Ts, **meta):
    """Write a human-readable .sol file.

    Keyword args in ``meta`` are written as ``key: value`` lines before the
    T-vector and route list (order is preserved in Python 3.7+).
    """
    with open(sol_path, "w") as fh:
        for k, v in meta.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"T1: {Ts[0]:.6f}  T2: {Ts[1]:.6f}  T3: {Ts[2]:.6f}\n")
        for i, r in enumerate(routes):
            fh.write(f"route {i + 1}: {' '.join(str(a) for a in r)}\n")
