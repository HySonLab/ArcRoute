import torch.nn as nn
import torch
class ARPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 4):
        super(ARPInitEmbedding, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(4, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        node_feats = torch.cat((            
                       td["demand"][..., None], 
                       td["clss"][..., 1:][..., None],
                       td["service_time"][..., 1:][..., None],
                       td["traveling_time"][..., 1:][..., None],
                       ), -1)
        
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_embeddings = self.init_embed(node_feats)

        depot_feats = torch.zeros_like(node_feats[:, 0:1, :])
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        depot_embedding = self.init_embed_depot(depot_feats)
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out