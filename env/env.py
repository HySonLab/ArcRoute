from tensordict.tensordict import TensorDict
import torch
import numpy as np
from env.generator import CARPGenerator
from common.cal_reward import calc_reward
from common.ops import gather_by_index, run_parallel
from torch.utils.data import DataLoader

class CARPEnv:
    def __init__(
        self,
        num_loc=60,
        num_arc=60,
        num_vehicle=3,
        variant= 'P',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = CARPGenerator
        self.num_loc = num_loc
        self.num_arc = num_arc
        self.num_vehicle = num_vehicle
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
        return DataLoader(
            self.generator(data_size, self.num_loc, self.num_arc, self.num_vehicle, data=data),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.generator.collate_fn
        )

    def get_objective(self, td, actions, local_search=True):
        actions = actions.clone().detach().cpu().numpy()
        td = td.clone().detach().cpu()
        return run_parallel(calc_reward, actions, td, num_workers=24, num_epochs=10, local_search=local_search)

    def get_reward(self, td, actions):
        actions = actions.clone().detach().cpu().numpy()
        td = td.clone().detach().cpu()
        rs = run_parallel(calc_reward, actions, td, num_workers=24, num_epochs=10, local_search=False, return_torch=True)
        rs = -rs[:, :1].to(td.device)
        # print(actions.shape, td.shape, rs.shape)
        return rs