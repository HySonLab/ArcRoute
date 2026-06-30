import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from utils.ops import gather_by_index


class ARPContext(nn.Module):
    # Context dim breakdown (7 scalars appended to current-node embedding):
    #   remaining_cap       (1) — vehicle_capacity - used_capacity
    #   frac_served_cls1/2/3(3) — clss_served / clss_total per class
    #   dist_to_depot       (1) — adj[current_node, depot=0]
    #   M_norm              (1) — num_vehicle / 10.0
    #   frac_cap_used       (1) — used_capacity / vehicle_capacity
    _STATE_DIM = 7

    def __init__(self, embed_dim, linear_bias=False):
        super(ARPContext, self).__init__()
        self.project_context = nn.Linear(
            embed_dim + self._STATE_DIM, embed_dim, bias=linear_bias
        )

    def _cur_node_embedding(self, embeddings, td):
        return gather_by_index(embeddings, td["current_node"])

    def _state_embedding(self, embeddings, td):
        B = td["used_capacity"].shape[0]
        dev = td["used_capacity"].device

        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]       # (B,1)

        total = td["clss_total"].clamp(min=1.0)                            # (B,3)
        frac_served = td["clss_served"] / total                            # (B,3)

        adj = td["adj"]                                                    # (B,N+1,N+1)
        cn  = td["current_node"].view(B)                                   # (B,)
        dist_to_depot = adj[torch.arange(B, device=dev), cn, 0].unsqueeze(-1)  # (B,1)

        M_norm = td["num_vehicle"].float().view(B, 1) / 10.0              # (B,1)

        frac_cap_used = td["used_capacity"] / td["vehicle_capacity"].clamp(min=1e-6)  # (B,1)

        return torch.cat(
            [remaining_cap, frac_served, dist_to_depot, M_norm, frac_cap_used], dim=-1
        )  # (B, 7)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        if cur_node_embedding.ndim == 1:
            cur_node_embedding = cur_node_embedding[None, :]
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class JointMVContext(nn.Module):
    """Vehicle state embedding for the Joint M×N decode (GRPO v3).

    For each of the M vehicles, build a query from
        [H[current_node_m], remaining_cap_m, completion_m_norm, dist_to_depot_m]
    and project it to the embedding dim. Produces (B, M, D) vehicle queries that
    the PointerAttention scores against the N+1 arc keys to give (B, M, N+1)
    logits — so the policy chooses (vehicle, arc) jointly at each step.
    """

    _VEH_STATE_DIM = 3  # remaining_cap, completion_norm, dist_to_depot

    def __init__(self, embed_dim, linear_bias=False):
        super().__init__()
        self.project_vehicle = nn.Linear(
            embed_dim + self._VEH_STATE_DIM, embed_dim, bias=linear_bias
        )

    def forward(self, node_embeddings, td):
        """node_embeddings: (B, N+1, D); td has per-vehicle current_node (B,M),
        used_capacity (B,M), completion (B,M) and adj (B,N+1,N+1).
        Returns (B, M, D) vehicle queries."""
        B, M = td["current_node"].shape
        dev = node_embeddings.device
        ar = torch.arange(B, device=dev)

        # Relative completion across this instance's vehicles (scale-invariant).
        comp_norm = td["completion"].amax(dim=-1, keepdim=True).clamp(min=1e-6)  # (B,1)

        veh_embeds = []
        for m in range(M):
            cn_m = td["current_node"][:, m]                  # (B,)
            h_m = node_embeddings[ar, cn_m]                  # (B, D)
            rem_m = (1.0 - td["used_capacity"][:, m]).unsqueeze(-1)            # (B,1)
            comp_m = (td["completion"][:, m:m + 1] / comp_norm)               # (B,1)
            dist_m = td["adj"][ar, cn_m, 0].unsqueeze(-1)                      # (B,1)
            veh_embeds.append(torch.cat([h_m, rem_m, comp_m, dist_m], dim=-1))  # (B,D+3)

        veh_stack = torch.stack(veh_embeds, dim=1)           # (B, M, D+3)
        return self.project_vehicle(veh_stack)               # (B, M, D)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class PointerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
        sdpa_fn=None,
        **kwargs,
    ):
        super(PointerAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.sdpa_fn = scaled_dot_product_attention
        self.check_nan = check_nan

    def forward(self, query, key, value, logit_key, attn_mask=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)   # (B, M, D)
        glimpse = self._project_out(heads, attn_mask)           # (B, M, D)

        # Logits per (query m, arc j): (B, M, D) x (B, D, N+1) -> (B, M, N+1).
        # The M (query) dimension is preserved so the Joint M×N decoder gets one
        # row of arc-logits per vehicle. With M==1 this is (B, 1, N+1).
        logits = torch.bmm(
            glimpse, logit_key.transpose(-2, -1)
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        return self.project_out(out)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        self.env_name = "tsp"
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        # ARPContext (single virtual vehicle, v2) is kept for backward compat but
        # is NOT used by the Joint M×N decode path. The joint path uses
        # JointMVContext to build one query per vehicle -> (B, M, D).
        self.context_embedding = ARPContext(embed_dim)
        self.joint_mv_context = JointMVContext(embed_dim)
        self.dynamic_embedding = StaticEmbedding(embed_dim)
        self.is_dynamic_embedding = False

        self.pointer = PointerAttention(
            embed_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            moe_kwargs=moe_kwargs,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        return glimpse_k, glimpse_v, logit_k

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
    ) -> Tuple[Tensor, Tensor]:
        """Joint M×N decode step.

        Returns:
            logits: (B, M, N+1) — arc logits per vehicle.
            mask:   (B, M, N+1) bool feasibility mask (True == feasible).
        """
        # Vehicle queries: (B, M, D).
        veh_q = self.joint_mv_context(cached.node_embeddings, td)
        graph_ctx = cached.graph_context                     # (B, D) or 0
        if isinstance(graph_ctx, Tensor):
            veh_q = veh_q + graph_ctx.unsqueeze(1)           # broadcast over M

        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Un-flatten the stored flat action_mask (B, M*(N+1)) -> (B, M, N+1).
        B = veh_q.shape[0]
        M = veh_q.shape[1]
        N1 = cached.node_embeddings.shape[1]
        mask = td["action_mask"].reshape(B, M, N1)           # True == feasible

        logits = self.pointer(veh_q, glimpse_k, glimpse_v, logit_k, mask)
        return logits, mask

    def pre_decoder_hook(self, td, env, embeddings):
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings)

    def post_decoder_hook(self, td, env):
        logprobs = torch.stack(self.logprobs, 1)
        actions = torch.stack(self.actions, 1)
        return logprobs, actions, td, env

    def _precompute_cache(self, embeddings: torch.Tensor) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )
