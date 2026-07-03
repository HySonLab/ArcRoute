from tensordict.tensordict import TensorDict
import torch
import numpy as np
from env.generator import SizeBucketBatchSampler
from utils.ops import gather_by_index
from torch.utils.data import DataLoader


class CARPEnv:
    def __init__(
        self,
        num_vehicle=3,
        variant='P',
        obj_weights=None,
        reward_mode='scalar',
        P=3,
    ):
        self.num_vehicle = num_vehicle
        self.variant = variant
        self.P = P
        self.reward_mode = reward_mode
        self.obj_weights = obj_weights if obj_weights is not None else [100.0 ** (-i) for i in range(self.P)]

    def step(self, td: TensorDict):
        """Joint M×N step. `td["action"]` is a flat joint index in [0, M_max*(N+1)):
        action = m_star * (N+1) + j_star, splitting into the chosen vehicle m_star
        and the chosen arc/node j_star."""
        dev = td.device
        B = td["used_capacity"].shape[0]
        N1 = td["visited"].shape[-1]                          # N+1
        ar = torch.arange(B, device=dev)

        action_flat = td["action"].long()
        m_star = action_flat // N1                            # (B,)
        j_star = action_flat % N1                             # (B,)
        is_depot = (j_star == 0)                              # (B,)

        dem_j = gather_by_index(td["demand"], j_star[:, None], squeeze=True)  # (B,)

        # step cost of vehicle m_star MUST use its OLD current_node (before update).
        old_cn_m = td["current_node"][ar, m_star]            # (B,)
        svc_j = gather_by_index(td["service_times"], j_star[:, None], squeeze=True)  # (B,)
        step_cost = td["adj"][ar, old_cn_m, j_star] + svc_j  # (B,)

        # Scatter-update only the selected vehicle slot — no full clone.
        # capacity: add demand, reset to 0 on depot return
        cap_delta = (td["used_capacity"][ar, m_star] + dem_j) * (~is_depot).float() \
                    - td["used_capacity"][ar, m_star]
        new_cap = td["used_capacity"] + torch.zeros_like(td["used_capacity"]).scatter_(
            1, m_star[:, None], cap_delta[:, None])

        # completion: add step cost
        new_comp = td["completion"] + torch.zeros_like(td["completion"]).scatter_(
            1, m_star[:, None], step_cost[:, None])

        # completion_at_arc[b, j] = new_comp[b, m] after serving arc j.
        # Depot visits (j_star==0) contribute delta=0 so depot slot stays 0.
        comp_m = new_comp[ar, m_star]                         # (B,) new vehicle time
        arc_delta = (comp_m - td["completion_at_arc"][ar, j_star]) * (~is_depot).float()
        new_comp_at_arc = td["completion_at_arc"] + \
            torch.zeros_like(td["completion_at_arc"]).scatter_(
                1, j_star[:, None], arc_delta[:, None])

        # current node: update to j_star
        new_cn = td["current_node"] + torch.zeros_like(td["current_node"]).scatter_(
            1, m_star[:, None], (j_star - td["current_node"][ar, m_star])[:, None])

        # Global served mask (depot is never marked served).
        visited = td["visited"].scatter(-1, j_star[:, None, None], 1)
        visited = torch.where(is_depot.view(B, 1, 1), td["visited"], visited)

        # done = all N non-depot arcs served.
        all_served = (visited[..., 0, 1:] != 0).all(dim=-1)   # (B,)
        done = all_served.view(B, 1)

        td.update(
            {
                "current_node": new_cn,
                "used_capacity": new_cap,
                "completion": new_comp,
                "completion_at_arc": new_comp_at_arc,
                "visited": visited,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        td.set("phase", self._phase(td))
        return td

    def _phase(self, td: TensorDict) -> torch.Tensor:
        """(B, 1) long — current phase = min still-unserved class (same c_min the
        precedence mask uses). Clamped to [1, P] so done rows stay a valid class
        index. Variant U has no precedence, so phase is a constant placeholder 1."""
        B = td["visited"].shape[0]
        if self.variant != 'P':
            return torch.ones(B, 1, dtype=torch.long, device=td.device)
        served_arcs = td["visited"][..., 0, 1:] != 0           # (B, N)
        clss_cust = td["clss"][..., 1:]                        # (B, N)
        masked_cls = clss_cust.masked_fill(served_arcs, 1 << 30)
        c_min = masked_cls.min(dim=-1, keepdim=True).values    # (B, 1)
        return c_min.clamp(1, self.P).long()

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Return a flat joint feasibility mask (B, M_max*(N+1)). True == FEASIBLE.
        The 3-layer mask (served | capacity(m,j) | precedence) plus the depot
        idle-guard and the nonexistent-vehicle guard are built on (B, M_max, N+1)
        and flattened."""
        dev = td.device
        B, _, N1 = td["visited"].shape                       # N1 = N+1
        M_max = td["used_capacity"].shape[-1]
        cap = 1.0

        served = td["visited"][..., 0, :]                    # (B, N+1) global served
        served_arcs = served[:, 1:] != 0                     # (B, N) bool
        all_served = served_arcs.all(dim=-1)                 # (B,) instance done

        # Layer 1 — served arcs (broadcast over M); depot col never served.
        served_col = (served != 0).clone()                   # (B, N+1) bool
        served_col[:, 0] = False

        # Layer 3 — precedence (variant P): block arcs of class > current min class.
        wrong_class = torch.zeros(B, N1, dtype=torch.bool, device=dev)
        if self.variant == 'P':
            clss_cust = td['clss'][..., 1:]                   # (B, N)
            unserved = ~served_arcs                           # (B, N) bool
            masked_cls = clss_cust.masked_fill(~unserved, 1 << 30)
            c_min = masked_cls.min(dim=-1, keepdim=True).values   # (B,1)
            wrong_class[:, 1:] = (clss_cust > c_min)

        # arc-level mask broadcast over vehicles: (B, 1, N+1)
        arc_mask = (served_col | wrong_class)[:, None, :]

        # Layer 2 — capacity per (m, j): demand[j] + used_cap[m] > cap.
        dem = td["demand"]                                   # (B, N+1)
        used = td["used_capacity"]                           # (B, M_max)
        exceeds_cap = (dem[:, None, :] + used[:, :, None]) > (cap + 1e-9)  # (B,M,N+1)
        # depot col: demand[0]=0 so never exceeds; zero it without clone via masking
        exceeds_cap = exceeds_cap & (torch.arange(N1, device=dev) > 0)[None, None, :]

        # Depot idle-guard: a vehicle AT the depot with EMPTY load cannot pick depot
        # again (prevents standing still). When the instance is done, never block
        # depot so done rows always keep ≥1 feasible cell (a vehicle can park).
        at_depot = (td["current_node"] == 0)                 # (B, M_max)
        is_empty = (td["used_capacity"] == 0)                # (B, M_max)
        block_depot = at_depot & is_empty & (~all_served[:, None])  # (B, M_max)
        depot_mask = torch.zeros(B, M_max, N1, dtype=torch.bool, device=dev)
        depot_mask[:, :, 0] = block_depot

        # Nonexistent-vehicle guard: vehicle m >= M_i does not exist -> fully masked.
        M_per_inst = td["num_vehicle"].reshape(B).to(dev)
        veh_range = torch.arange(M_max, device=dev).view(1, M_max)
        nonexistent = (veh_range >= M_per_inst.view(B, 1))    # (B, M_max)
        nonexistent_mask = nonexistent[:, :, None].expand(B, M_max, N1)

        full_mask = arc_mask | exceeds_cap | depot_mask | nonexistent_mask  # (B,M,N+1)
        flat_mask = full_mask.reshape(B, M_max * N1)
        return ~flat_mask                                    # True == feasible

    def reset(self, td):
        batch_size = td.batch_size[0]
        dev = td.device

        # Per-instance fleet size M_i and the padded fleet dim M_max.
        nv = torch.as_tensor(td["num_vehicle"], device=dev).reshape(batch_size).long()
        M_max = int(nv.max().item())

        N1 = td["service_times"].shape[1]  # N+1 (depot + N arcs)
        td_reset = TensorDict(
            {
                "demand": td["demands"],
                # Per-vehicle state: (B, M_max).
                "current_node": torch.zeros(
                    batch_size, M_max, dtype=torch.long, device=dev),
                "used_capacity": torch.zeros(batch_size, M_max, device=dev),
                "completion": torch.zeros(batch_size, M_max, device=dev),
                # completion_at_arc[b, j] = absolute time when arc j was served.
                # Updated in step(); used by get_reward_gpu() to compute T_k on GPU
                # (eliminates CPU Scheduler from the training loop).
                "completion_at_arc": torch.zeros(batch_size, N1, device=dev),
                "vehicle_capacity": torch.ones(batch_size, 1, device=dev),
                "visited": torch.zeros(
                    batch_size, 1, N1,
                    dtype=torch.uint8, device=dev,
                ),
                'clss': td["clss"],
                'service_times': td["service_times"],
                'traversal_times': td["traversal_times"],
                'adj': td["adj"],
                "num_vehicle": nv.reshape(batch_size, 1),
                "done": torch.zeros(batch_size, 1, dtype=torch.bool, device=dev),
            },
            batch_size=batch_size,
        ).to(dev)
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        td_reset.set("phase", self._phase(td_reset))
        return td_reset
    
    
    def dataset(self, data_dir, batch_size=128, shuffle=False, buckets=None):
        """Build a DataLoader from an OSM data directory.

        data_dir: directory produced by scripts/gen_data.sh.
        buckets:  list of bucket keys to load, e.g. ["20_30", "20_40"]
                  (from utils.data_utils.CURRICULUM). None = all buckets.
        Cache key includes bucket names so different phases don't share a cache.
        """
        from env.osm_dataset import OSMMultiSizeDataset
        choices = self.num_vehicle if isinstance(self.num_vehicle, list) else [self.num_vehicle]
        suffix = ("_" + "_".join(sorted(buckets))) if buckets else ""
        cache = data_dir.rstrip("/") + suffix + ".cache.pt"
        ds = OSMMultiSizeDataset(data_dir, vehicle_choices=choices,
                                 cache_path=cache, buckets=buckets)
        sampler = SizeBucketBatchSampler(ds.bucket_ranges, batch_size, shuffle=shuffle)
        return DataLoader(ds, batch_sampler=sampler, num_workers=4,
                          collate_fn=ds.collate_fn, pin_memory=True,
                          persistent_workers=True)

    def _pos_val(self):
        return list(range(1, self.P + 1))

    def get_reward(self, td, actions=None):
        """Compute T_k (B, P) on GPU from completion_at_arc tracked in env.step().

        completion_at_arc[b, j] = absolute time when arc j was served by its vehicle.
        T_k = max over class-k arcs — no CPU Scheduler, no D2H transfer."""
        comp = td["completion_at_arc"]   # (B, N+1)
        clss = td["clss"]                # (B, N+1)
        B = comp.shape[0]
        dev = comp.device
        T = torch.zeros(B, self.P, device=dev, dtype=comp.dtype)
        for ki, k in enumerate(self._pos_val()):
            mask_k = (clss == k).float()
            T[:, ki] = (comp * mask_k).max(dim=-1).values
        if self.reward_mode == 'vector':
            return T
        w = torch.tensor(self.obj_weights, dtype=T.dtype, device=dev)
        return -(T * w).sum(-1, keepdim=True)