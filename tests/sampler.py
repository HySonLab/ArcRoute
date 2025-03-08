import torch
from torch.distributions import Uniform, Normal
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

def get_sampler(
    distribution: str,
    low: float = 0.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    **kwargs,
):
    if distribution == "uniform":
        return Uniform(low, high)
    elif distribution == "normal":
        return Normal(mean, std)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def sample_arcs(num_loc, num_arc):
    coms = torch.combinations(torch.arange(num_loc), r=2)
    coms = torch.stack([*coms, *coms[:, [1, 0]]])
    
    idxs = torch.randperm(coms.size(0))[:num_arc+1]
    arcs = coms[idxs]
    arcs[0] = torch.Tensor([0, 0])
    return arcs

def sample_traversal_time(num_loc, arcs, coord_sampler):
    coords = coord_sampler.sample((num_loc, 2))
    dists = torch.cdist(coords, coords, p=2)
    traversal_time = dists[arcs[:, 0], arcs[:, 1]].clone()

    for k in range(num_loc):
        dists = torch.min(dists, dists[:, k].unsqueeze(1) + dists[k, :].unsqueeze(0))

    dists_edges = dists[arcs[..., 1].unsqueeze(-1), arcs[..., 0].unsqueeze(0)]
    return traversal_time, dists_edges

def sample_service_time(traversal_time):
    return traversal_time * 2

def sample_demand(traversal_time, clss):
    demand = traversal_time * 0.5 + 0.5
    demand[clss] = 0
    return demand

def sample_vehicle_capacity(traversal_time, priority_classes):
    return (traversal_time[priority_classes]/3 + 0.5).sum()

def sample_priority_classes(num_arc):
    priority_classes = torch.randint(1, 3+1, size=(num_arc+1, ), dtype=torch.int64)

    ids_0 = torch.randperm(num_arc)[:int(num_arc*25/100)+1]
    if num_arc > 80:
        ids_0 = torch.randperm(num_arc)[:num_arc - torch.randint(60, 70, (1, ))+1]
    ids_0[0] = 0
    priority_classes[ids_0] = 0

    return priority_classes

def generate(num_loc, num_arc):
    coord_sampler = get_sampler("uniform", low=0, high=1)
    arcs = sample_arcs(num_loc, num_arc)
    clss = sample_priority_classes(num_arc)
    traversal_time, dists_edges = sample_traversal_time(num_loc, arcs, coord_sampler)
    servicing_time = sample_service_time(traversal_time)
    demands = sample_demand(traversal_time, clss)
    vehicle_capacity = sample_vehicle_capacity(demands, clss)
    td = TensorDict(
            {
                'clss': clss,
                "demand": demands / vehicle_capacity,
                "capacity": vehicle_capacity,
                "service_time": servicing_time,
                "traversal_time": traversal_time,
                "adj": dists_edges
            },
        )
    td = td.unsqueeze(0)
    td.batch_size=torch.Size([1]) 
    return td

class CARPGenerator(Dataset):
    def __init__(self, num_samples, num_loc, num_arc, cache_file="carp_data.pt"):

        self.num_samples = num_samples
        self.num_loc = num_loc
        self.num_arc = num_arc
        self.cache_file = cache_file

        # Creating dataset ...
        if os.path.exists(self.cache_file):
            print(f"Loading dataset from {self.cache_file}...")
            self.data = torch.load(self.cache_file)
        else:
            print("Generating new dataset...")
            self.data = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        if len(self.data) < self.num_samples:
            self.data.append(generate(self.num_loc, self.num_arc))
            return self.data[-1]
        
        return self.data[idx]
    

def collate_fn(batch):
    return torch.cat(batch, dim=0) 

def save_cache(cache_file):

    dataloader = DataLoader(
        CARPGenerator(100000, 60, 60),
        batch_size=128,
        shuffle=False,
        num_workers=24,
        collate_fn=collate_fn,
    )
    tds = []
    for td in tqdm(dataloader):
        tds.append(td)
    
    # tds = torch.cat(tds, dim=0)

    print(f"Saving dataset to {cache_file}...")
    torch.save(tds, cache_file)
    
if __name__ == "__main__":
    torch.manual_seed(10)

    # data = torch.load("carp_data.pt")
    # print(data)
    # save_cache("carp_data.pt")
    dataloader = DataLoader(
        CARPGenerator(100000, 60, 60),
        batch_size=128,
        shuffle=False,
        num_workers=24,
        collate_fn=collate_fn,
    )

    for i in range(3):
        for td in tqdm(dataloader):
            pass

    
    
    