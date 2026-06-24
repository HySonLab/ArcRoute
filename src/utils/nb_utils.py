import numpy as np


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
    # Trim each row to actual_max (rows are equal length already, but trim
    # trailing all-zero padding columns shared by every row).
    out = np.zeros((len(rows), actual_max), dtype=np.int32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r[:actual_max]
    return out


def convert_prob(x):
    """Softmax via log-sum-exp normalisation."""
    x = np.asarray(x, dtype=np.float64)
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)
