from torch.distributions import Uniform
import torch
from torch.utils.data import Dataset
from tensordict.tensordict import TensorDict
from common.ops import gather_by_index
import numpy as np
from einops import repeat

def get_sampler(
    val_name: str,
    distribution,
    low: float = 0,
    high: float = 1.0,
    **kwargs,
):
    if isinstance(distribution, (int, float)):
        return Uniform(low=distribution, high=distribution)
    elif distribution == Uniform or distribution == "uniform":
        return Uniform(low=low, high=high)

CAPACITIES = {10: 20.0, 15: 25.0, 20: 30.0, 30: 33.0, 40: 37.0, 50: 40.0, 60: 43.0, 75: 45.0, 
    100: 50.0, 125: 55.0, 150: 60.0, 200: 70.0, 500: 100.0, 1000: 150.0}

class CARPGenerator:
    def __init__(
        self,
        num_loc: int = 20,
        num_arc: int = 10,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution= Uniform,
        depot_distribution = Uniform,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution = Uniform,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        **kwargs
    ):
        self.num_loc = num_loc
        self.num_arc = num_arc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler("loc", loc_distribution, min_loc, max_loc, **kwargs)

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler("demand", demand_distribution, min_demand-1, max_demand-1, **kwargs)
        
        # Capacity
        self.capacity = kwargs.get("capacity", None)
        if self.capacity is None:
            self.capacity = CAPACITIES.get(num_arc, None)
        if self.capacity is None:
            closest_num_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_arc))
            self.capacity = CAPACITIES[closest_num_loc]
        
        self.capacity = self.capacity * 2

    def __call__(self, batch_size):
        # Sample arc demands and define arcs as pairs of nodes
        combinations = torch.combinations(torch.arange(self.num_loc), r=2)
        idxs = np.arange(combinations.size(0))
        idxs = np.stack([np.random.choice(idxs, size=self.num_arc, replace=False) for _ in range(batch_size)])
        idxs = torch.tensor(idxs)
        combinations = repeat(combinations, 'n d -> b n d', b=batch_size, d=2)
        arc_indices = gather_by_index(combinations, idxs, squeeze=False)
        demands = self.demand_sampler.sample((batch_size, self.num_arc))
        demands = (demands.int() + 1).float()
        demands = torch.cat((torch.zeros(batch_size, 1), demands), dim=-1)
        arc0 = torch.zeros(batch_size*2).reshape(batch_size, 1, 2)
        arc_indices = torch.cat([arc0, arc_indices], dim=-2).type(torch.int32)
        # Sample capacities
        capacity = torch.full((batch_size, 1), self.capacity)
        clss = torch.randint(1, 3+1, size=(batch_size, self.num_arc))
        clss = torch.cat([torch.zeros(batch_size, 1), clss], dim=-1)

        a, b = 1, 2
        service_time = a + (b - a) * torch.rand(batch_size, self.num_arc)
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)

        a, b = 0, 4
        dms = a + (b - a) * torch.rand(batch_size, self.num_loc, self.num_loc)
        bs, n_nodes, _ = dms.size()
        dms[dms == 0] = float('inf')
        torch.diagonal(dms, dim1=1, dim2=2).fill_(0)
        dms[torch.arange(bs).view(bs, 1), arc_indices[..., 0], arc_indices[..., 1]] = torch.rand_like(service_time)
        for k in range(n_nodes):
            dms = torch.minimum(dms, dms[:, :, k].view(bs, -1, 1) + dms[:, k, :].view(bs, 1, -1))
        
        traveling_time = dms[torch.arange(bs).view(bs, 1), arc_indices[..., 0], arc_indices[..., 1]]
        
        start_nodes = arc_indices[..., 0]  # Shape: (bs, num_arc+1)
        end_nodes = arc_indices[..., 1]    # Shape: (bs, num_arc+1)
        adj = dms[torch.arange(bs).unsqueeze(1).unsqueeze(2), end_nodes.unsqueeze(-1), start_nodes.unsqueeze(1)]
        return TensorDict(
            {
                'clss': clss.type(torch.int64),
                "demand": demands / self.capacity,
                "capacity": capacity,
                "service_time": service_time,
                "traveling_time": traveling_time,
                "adj": adj
            },
            batch_size=batch_size,
        )

    
class TensorDictDataset(Dataset):
    def __init__(self, td: TensorDict):
        self.data_len = td.batch_size
        if not isinstance(self.data_len, int):
            self.data_len = self.data_len[0]
        self.data = [
            {key: value[i] for key, value in td.items()} for i in range(self.data_len)
        ]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]

    def add_key(self, key, value):
        return ExtraKeyDataset(self, value, key_name=key)

    @staticmethod
    def collate_fn(batch):
        """Collate function compatible with TensorDicts that reassembles a list of dicts."""
        return TensorDict(
            {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
            batch_size=torch.Size([len(batch)]))
    
class ExtraKeyDataset(TensorDictDataset):

    def __init__(self, dataset: TensorDictDataset, extra: torch.Tensor, key_name="extra"):
        self.data_len = len(dataset)
        assert self.data_len == len(extra), "Data and extra must be same length"
        self.data = dataset.data
        self.extra = extra
        self.key_name = key_name

    def __getitem__(self, idx):
        data = self.data[idx]
        data[self.key_name] = self.extra[idx]
        return data