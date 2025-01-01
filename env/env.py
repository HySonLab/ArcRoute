import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensordict.tensordict import TensorDict
import torch
import numpy as np
from env.generator import CARPGenerator, TensorDictDataset
from common.cal_reward import get_reward, get_Ts_RL
from common.local_search import lsRL
from common.nb_utils import gen_tours_batch
from common.ops import gather_by_index

class CARPEnv:
    def __init__(
        self,
        generator_params: dict = {},
        variant= 'P',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = CARPGenerator(**generator_params)
        self.dataset_cls = TensorDictDataset
        self.variant = variant

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
        # print(td["demand"][:, None, :].shape, td["used_capacity"][..., None].shape, td["vehicle_capacity"][..., None].shape)
        # exit()
        exceeds_cap = (
            td["demand"][..., 1:][:, None, :] + td["used_capacity"][..., None] > td["vehicle_capacity"][..., None]
        )
        # print(exceeds_cap.shape)
        # exit()
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap
        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        # print(mask_loc.shape, mask_depot.shape)
        # exit()
        if self.variant == 'P':
            clss_min = gather_by_index(td['clss'], td["current_node"])
            mask_loc = mask_loc | ((td['clss'][..., 1:] - clss_min[..., None] < 0).unsqueeze(1))
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

    def reset(self, td: TensorDict = None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
            if not isinstance(batch_size, int):
                batch_size = batch_size[0]        
        
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "demand": td["demand"],
                "current_node": torch.zeros(
                    batch_size, 1, dtype=torch.long),
                "used_capacity": torch.zeros((batch_size, 1)),
                "vehicle_capacity": torch.full(
                    (batch_size, 1), self.generator.vehicle_capacity),
                "visited": torch.zeros(
                    (batch_size, 1, td["service_time"].shape[1]),
                    dtype=torch.uint8
                ),
                'clss': td["clss"],
                'service_time': td["service_time"],
                'traveling_time': td["traveling_time"],
                'adj': td["adj"],
                "done": torch.zeros(batch_size, td["service_time"].shape[1], dtype=torch.bool),
            },
            batch_size=batch_size,
        ).to(td.device)
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    
    def dataset(self, batch_size, phase=None):
        td = self.generator(batch_size)
        return self.dataset_cls(td)

    def get_objective(self, td, actions, is_local_search=True):
        tours_batch = gen_tours_batch(actions.cpu().numpy().astype(np.int32))
        if is_local_search:
            tours_batch = lsRL(td, tours_batch=tours_batch, variant=self.variant, is_train=False)  
        return get_Ts_RL(td, tours_batch=tours_batch)

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        tours_batch = gen_tours_batch(actions.cpu().numpy().astype(np.int32))
        tours_batch = lsRL(td, tours_batch=tours_batch, variant=self.variant, is_train=True)
        r = get_reward(td, tours_batch=tours_batch)
        r = -torch.tensor(r, device=td.device)
        return r