import numpy as np
import concurrent.futures

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

    
def run_parallel(operation, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(operation, *param_set, **kwargs) for param_set in zip(*args)]
        return [f.result() for f in futures]

def run_parallel2(operation, *args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(50) as executor:
        futures = [executor.submit(operation, *param_set, **kwargs) for param_set in zip(*args)]
    
    return [f.result() for f in futures]

def convert_vars_np(td):
    import torch
    adj = td['adj'].detach().clone()
    torch.diagonal(adj, dim1=1, dim2=2).fill_(float('inf'))
    return {
        'adj': adj.cpu().numpy(),
        'service_time': td['service_time'].detach().cpu().numpy(),
        'clss': td['clss'].detach().cpu().numpy().astype(np.int32),
        'demand': td['demand'].detach().cpu().numpy()
    }


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

def import_instance(es):
    if isinstance(es, str):
        es = np.load(es)
    C = es['C']
    P = [i for i in range(1, es['P']+1)]
    M = [i for i in range(es['M'])]
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
