import numpy as np
import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    td = torch.randn(128)
    idxs = torch.randperm(128)
    batch_size = 10
    for i in range(0, 128, batch_size):
        id_sub = idxs[i:i+batch_size]
        print(td[id_sub])