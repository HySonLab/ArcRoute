from tensordict.tensordict import TensorDict
import torch
import numpy as np
from env.generator import (CARPGenerator, MultiSizeCARPGenerator,
                           SizeBucketBatchSampler)
from solvers.cal_reward import calc_reward
from utils.ops import gather_by_index, run_parallel
from torch.utils.data import DataLoader


def reward_num_workers(n):
    """Worker count for the reward/objective DataLoader, profiled on the Scheduler
    (sub-millisecond numpy tasks). The old fixed `num_workers=24` was the WORST
    choice at every batch size: forking 24 procs + pickling TensorDict slices costs
    more than the compute. Below ~1k tasks inline (0 workers, main process) wins;
    above it a modest pool (8) amortizes the fork overhead. Output is identical
    regardless of worker count (DataLoader keeps order; calc_reward is pure)."""
    return 0 if n < 1024 else 8


class CARPEnv:
    def __init__(
        self,
        num_loc=60,
        num_arc=60,
        num_vehicle=3,
        variant= 'P',
        sizes=None,
        obj_weights=None,
        reward_mode='scalar',
        P=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = CARPGenerator
        self.num_loc = num_loc
        self.num_arc = num_arc
        self.num_vehicle = num_vehicle
        self.variant = variant
        # Number of priority classes (objective is lexicographic T_1 <= ... <= T_P).
        # Default 3 = the paper's setting. P flows to the generator (class sampling)
        # and to the reward (Scheduler pos_val = 1..P -> a (B,P) T-vector).
        self.P = P
        # D2 Phase 2: 'scalar' (default) keeps the old -(T . w) reward byte-identical;
        # 'vector' returns the raw (B,3) T-vector so the GRPO path (Phase 3) can rank
        # the K samples lexicographically instead of scalarizing.
        self.reward_mode = reward_mode
        # Phase 6: if `sizes` (list of (num_loc, num_arc)) is given, train over a
        # size LADDER (bucketed so each batch stays single-size). Else single-size.
        self.sizes = sizes
        # Hierarchical objective weights (T_1, T_2, T_3). The RL reward is -(T . w),
        # giving T_1 priority (w_1 >> w_2 >> w_3) with T_2/T_3 breaking ties.
        # Default is PROPORTIONAL to baseline/meta.py:calc_obj's [1e3,1e1,1e-1]
        # (same 100x ratios -> identical solution ranking) but scaled to ~O(T_1)
        # magnitude so PPO's critic stays well-conditioned (the absolute scale is
        # irrelevant: advantages are normalised, and eval reports T_1/T_2/T_3
        # separately). Previously the reward was -T_1 only (T_2/T_3 got no signal).
        # Weighted-sum weights (PPO path) -- one per class, geometrically separated
        # (100x) so T_1 dominates. Default generalizes the paper's [1.0,1e-2,1e-4]
        # to any P; P=3 reproduces it exactly.
        self.obj_weights = obj_weights if obj_weights is not None else [100.0 ** (-i) for i in range(self.P)]

    def step(self, td: TensorDict):
        current_node = td["action"][:, None]  # Add dimension for step

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], current_node, squeeze=False)

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        visited = td["visited"].scatter(-1, current_node[..., None], 1)
        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (
            td["demand"][..., 1:][:, None, :] + td["used_capacity"][..., None] > td["vehicle_capacity"][..., None]
        )
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap
        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        if self.variant == 'P':
            # GLOBAL precedence (B+): serve ALL of the lowest unserved class before
            # any higher class -> the policy emits one coherent tour per class, and
            # the Scheduler spreads each class across the M vehicles. (The previous
            # floor was the CURRENT node's class, which reset to 0 at every depot
            # return -> only per-route ordering, which the capacity re-split then
            # broke; see common/scheduler.py.)
            clss_cust = td['clss'][..., 1:]                              # (B, n)
            unserved = td['visited'][..., 1:].squeeze(1) == 0           # (B, n) bool
            masked = clss_cust.masked_fill(~unserved, 1 << 30)
            c_min = masked.min(dim=-1, keepdim=True).values            # (B,1) lowest unserved class
            mask_loc = mask_loc | ((clss_cust > c_min).unsqueeze(1))
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

    def reset(self, td):
        batch_size = td.batch_size[0]      
        td_reset = TensorDict(
            {
                "demand": td["demands"],
                "current_node": torch.zeros(
                    batch_size, 1, dtype=torch.long),
                "used_capacity": torch.zeros((batch_size, 1)),
                "vehicle_capacity": torch.full((batch_size, 1), 1),
                "visited": torch.zeros(
                    (batch_size, 1, td["service_times"].shape[1]),
                    dtype=torch.uint8
                ),
                'clss': td["clss"],
                'service_times': td["service_times"],
                'traversal_times': td["traversal_times"],
                'adj': td["adj"],
                # M-agnostic policy doesn't read this; the Scheduler (calc_reward)
                # does. Carry per-instance M as (B,1) so run_parallel slices it right.
                "num_vehicle": torch.as_tensor(td["num_vehicle"]).reshape(batch_size, 1),
                "done": torch.zeros(batch_size, td["service_times"].shape[1], dtype=torch.bool),
            },
            batch_size=batch_size,
        ).to(td.device)
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    
    def dataset(self, data_size,
                batch_size=128,
                shuffle=False,
                num_workers=24,
                data='carp_data.pt'):
        # Scale workers to data size: few workers for small datasets (fork overhead
        # dominates), more for large ones. Cap at the requested num_workers.
        effective_workers = min(num_workers, max(0, data_size // 2000))
        pin = effective_workers > 0
        if self.sizes is not None:
            # Multi-size: bucket by size, each batch single-size (Phase 6).
            ds = MultiSizeCARPGenerator(data_size, self.sizes, self.num_vehicle,
                                        num_workers=num_workers, data=data, P=self.P)
            sampler = SizeBucketBatchSampler(ds.bucket_ranges, batch_size, shuffle=shuffle)
            return DataLoader(ds, batch_sampler=sampler, num_workers=effective_workers,
                              collate_fn=ds.collate_fn, pin_memory=pin,
                              persistent_workers=effective_workers > 0)
        return DataLoader(
            self.generator(data_size, self.num_loc, self.num_arc, self.num_vehicle,
                           data=data, P=self.P),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_workers,
            collate_fn=self.generator.collate_fn,
            pin_memory=pin,
            persistent_workers=effective_workers > 0,
        )

    def _pos_val(self):
        """Priority class labels 1..P for the Scheduler/reward (-> a (B,P) T-vector)."""
        return list(range(1, self.P + 1))

    def get_objective(self, td, actions, local_search=True):
        actions = actions.clone().detach().cpu().numpy()
        td = td.clone().detach().cpu()
        return run_parallel(calc_reward, actions, td,
                            num_workers=reward_num_workers(len(actions)), num_epochs=10,
                            local_search=local_search, variant=self.variant,
                            pos_val=self._pos_val())

    def get_reward(self, td, actions):
        actions = actions.clone().detach().cpu().numpy()
        td = td.clone().detach().cpu()
        rs = run_parallel(calc_reward, actions, td,
                          num_workers=reward_num_workers(len(actions)), num_epochs=10,
                          local_search=False, return_torch=True, variant=self.variant,
                          pos_val=self._pos_val())  # (B,P) T-vector
        if self.reward_mode == 'vector':
            # GRPO path: hand the raw (B,3) T-vector (T positive, lower=better) to
            # the caller, which ranks the K samples lexicographically (Phase 3).
            return rs.to(td.device)
        # Hierarchical scalar reward = -(T . w): optimise T_1 first (w_1 >> w_2 >>
        # w_3), with T_2/T_3 breaking ties. Full T vector stays in get_objective.
        w = torch.tensor(self.obj_weights, dtype=rs.dtype, device=rs.device)
        rs = -(rs * w).sum(-1, keepdim=True).to(td.device)
        return rs