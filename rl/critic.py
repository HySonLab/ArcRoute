import copy
from typing import Union
from tensordict import TensorDict
from torch import Tensor, nn
import torch

class CriticNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        customized: bool = False,
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.value_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
        self.customized = customized

    def forward(self, x: Union[Tensor, TensorDict], hidden=None) -> Tensor:
        if not self.customized:  # fir for most of costructive tasks
            h, _ = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]
            o = self.value_head(h).mean(1)  # [batch_size, N] -> [batch_size]
        else:  # custimized encoder and value head with hidden input
            h = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]
            o = self.value_head(h, hidden)
        o = torch.clamp(o, -1e4, 1e4)
        return o

def create_critic_from_actor(policy, backbone='encoder'):
    encoder = getattr(policy, backbone, None)
    embed_dim = getattr(policy, 'embed_dim', None)
    critic = CriticNetwork(copy.deepcopy(encoder), embed_dim).to(
        next(policy.parameters()).device
    )
    return critic