from typing import Tuple, Union, List
from typing import Callable, Optional
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch import Tensor
from typing import Tuple, Union
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention
from .init import ARPInitEmbedding
    
class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
            }.get(normalization, None)

            self.normalizer = normalizer_class(embed_dim, affine=True)
        else:
            self.normalizer = "layer"

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: List[int] = [64, 32],
        dropout_probs: Union[None, List[float]] = None,
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "None",
        output_norm: str = "None",
    ):
        super(MLP, self).__init__()

        assert input_norm in ["Batch", "Layer", "None"]
        assert output_norm in ["Batch", "Layer", "None"]

        if dropout_probs is None:
            dropout_probs = [0.0] * len(num_neurons)
        elif len(dropout_probs) != len(num_neurons):
            dropout_probs = [0.0] * len(num_neurons)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()
        self.dropouts = []
        for i in range(len(dropout_probs)):
            self.dropouts.append(nn.Dropout(p=dropout_probs[i]))

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim))

        self.input_norm = self._get_norm_layer(input_norm, input_dim)
        self.output_norm = self._get_norm_layer(output_norm, output_dim)

    def forward(self, xs):
        xs = self.input_norm(xs)
        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
            xs = self.dropouts[i](xs)
        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = self.output_norm(xs)
        return xs

    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            in_norm = nn.BatchNorm1d(dim)
        elif norm_method == "Layer":
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            in_norm = nn.Identity()  # kinda placeholder
        else:
            raise RuntimeError(
                "Not implemented normalization layer type {}".format(norm_method)
            )
        return in_norm

    def _get_act(self, is_last):
        return self.out_act if is_last else self.hidden_act
    
            
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
    
class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        ffn = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")

        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(ffn),
            Normalization(embed_dim, normalization),
        )


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    moe_kwargs=moe_kwargs,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x)
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(Encoder, self).__init__()

        self.embed_dim = embed_dim

        self.init_embedding = ARPInitEmbedding(embed_dim)

        self.net = (
            GraphAttentionNetwork(
                num_heads,
                embed_dim,
                num_layers,
                normalization,
                feedforward_hidden,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        # Process embedding
        h = self.net(init_h, mask)

        # Return latent representation and initial embedding
        return h, init_h
