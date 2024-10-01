import torch.nn as nn
import torch
from common.ops import  gather_by_index

class ARPContext(nn.Module):
    def __init__(self, embed_dim, linear_bias=False):
        super(ARPContext, self).__init__()
        self.project_context = nn.Linear(embed_dim + 1, embed_dim, bias=linear_bias)
    
    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding
    
    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        if len(cur_node_embedding.shape) == 1:
            cur_node_embedding = cur_node_embedding[None, :]
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)