
def gen_tours(action):
    idxs = [0] + [i+1 for i in range(len(action)) if action[i] == 0] + [len(action)]
    tours = []
    maxlen = 0
    for i,j in zip(idxs[:-1], idxs[1:]):
        a = action[i:j]
        if a.sum() == 0:
            continue
        tours.append(a)
        maxlen = max(maxlen, len(a))
    padded = np.zeros((len(tours), maxlen+2), dtype=np.int32)
    for idx, tour in enumerate(tours):
        padded[idx][1:len(tour)+1] = tour
    return padded

@torch.jit.script
def action_to_tours2(action: torch.Tensor):
    # Ensure all tensors are on the same device as 'action'
    device = action.device
    split_indices = torch.cat((torch.tensor([-1], device=device), 
                               torch.where(action == 0)[0], 
                               torch.tensor([len(action)], device=device)))
    
    # Compute segment lengths
    lengths = torch.diff(split_indices) - 1
    
    # Filter valid lengths and convert to list
    valid_lengths = lengths[lengths > 0].int()  # Ensure int type
    split_sizes = [int(x.item()) for x in valid_lengths]  # Convert to Python list

    # Split the tensor
    result = torch.split(action[action != 0], split_sizes)
    
    return result  # Returns a tuple of tensors (TorchScript-compatible)
