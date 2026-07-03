import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


class JointMVContext(nn.Module):
    """Vehicle state embedding for the Joint M×N decode (GRPO v3, bậc-3 upgrade).

    For each of the M vehicles, build a query from
        [H[current_node_m], remaining_cap_m, completion_m_norm, dist_to_depot_m,
         phase/P, per-class remaining-arc-fraction (P scalars)]
    projected to the embedding dim ("individual" query h¹), then mix the M
    vehicle tokens with one masked self-attention layer ("collaborative" h²,
    RANS-style Dual Cross-Attention — real vehicles never attend to zero-state
    padding/ghost slots), and fuse: veh_q = Linear(concat(h¹, h²)).
    Produces (B, M, D) vehicle queries that the PointerAttention scores against
    the N+1 arc keys to give (B, M, N+1) logits — so the policy chooses
    (vehicle, arc) jointly at each step.
    """

    _BASE_VEH_STATE_DIM = 3  # remaining_cap, completion_norm, dist_to_depot

    def __init__(self, embed_dim, P=3, linear_bias=False):
        super().__init__()
        self.P = P
        # + 1 (phase/P) + P (per-class remaining-arc-fraction)
        self._VEH_STATE_DIM = self._BASE_VEH_STATE_DIM + 1 + P
        self.project_vehicle = nn.Linear(
            embed_dim + self._VEH_STATE_DIM, embed_dim, bias=linear_bias
        )
        # Self-attention among the M vehicle tokens (single-head — M <= ~10
        # tokens, so head-splitting buys nothing; same sdpa primitive as
        # PointerAttention._inner_mha).
        self.veh_attn_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=linear_bias)
        self.veh_attn_out = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        # Fuse individual + collaborative (RANS eq. 25 style, single linear).
        self.fuse = nn.Linear(2 * embed_dim, embed_dim, bias=linear_bias)

    def _class_progress(self, td, M, dtype):
        """Global (per-instance) endgame features, broadcast over vehicles.

        Returns:
            phase_feat  (B, M, 1): current phase / P (td['phase'] is (B,1) long).
            remain_frac (B, M, P): per class p in 1..P, fraction of class-p arcs
                                    still unserved (0 if the class is absent).
        """
        B = td["clss"].shape[0]
        dev = td["clss"].device
        phase_feat = (td["phase"].to(dtype) / self.P).reshape(B, 1, 1).expand(B, M, 1)

        clss = td["clss"][..., 1:]                           # (B, N) — skip depot col
        served = td["visited"][..., 0, 1:] != 0              # (B, N) bool
        p_range = torch.arange(1, self.P + 1, device=dev)    # (P,)
        is_p = clss.unsqueeze(-1) == p_range                 # (B, N, P)
        total_p = is_p.sum(dim=1)                            # (B, P)
        unserved_p = (is_p & ~served.unsqueeze(-1)).sum(dim=1)  # (B, P)
        remain_frac = unserved_p.to(dtype) / total_p.clamp(min=1).to(dtype)  # (B, P)
        return phase_feat, remain_frac.unsqueeze(1).expand(B, M, self.P)

    def forward(self, node_embeddings, td):
        """node_embeddings: (B, N+1, D); td has per-vehicle current_node (B,M),
        used_capacity (B,M), completion (B,M), adj (B,N+1,N+1), plus global
        phase (B,1), clss (B,N+1), visited (B,1,N+1), num_vehicle (B,1).
        Returns (B, M, D) vehicle queries. Fully vectorised — no Python loop over M."""
        B, M = td["current_node"].shape
        dev = node_embeddings.device
        cn = td["current_node"]                              # (B, M)

        # Real (non-padding) vehicle slots: (B, M) bool, True == real. Instances
        # with M_i < M_max carry zero-state ghost slots the env never selects.
        valid_veh = (
            torch.arange(M, device=dev).view(1, M) < td["num_vehicle"].reshape(B, 1)
        )

        # Node embedding for each vehicle's current node: (B, M, D)
        B_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, M)
        h = node_embeddings[B_idx, cn]                       # (B, M, D)

        # Remaining capacity: (B, M, 1)
        rem = (1.0 - td["used_capacity"]).unsqueeze(-1)

        # Normalised completion (scale-invariant across vehicles): (B, M, 1).
        # Max over REAL vehicles only (ghost completion is 0 anyway in the env,
        # so this is numerically identical — it just makes padding-invariance
        # exact by construction).
        comp_norm = (
            td["completion"].masked_fill(~valid_veh, 0.0)
            .amax(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        comp = (td["completion"] / comp_norm).unsqueeze(-1)  # (B, M, 1)

        # Distance to depot: adj[:, cn, 0] -> (B, M, 1), normalised per instance
        adj_scale = td["adj"].amax(dim=(-2, -1)).clamp(min=1e-6).unsqueeze(1)  # (B, 1)
        dist = (td["adj"][:, :, 0].gather(1, cn) / adj_scale).unsqueeze(-1)   # (B, M, 1)

        # Global phase + per-class endgame progress (same for all M vehicles).
        phase_feat, remain_frac = self._class_progress(td, M, node_embeddings.dtype)

        feat = torch.cat([h, rem, comp, dist, phase_feat, remain_frac], dim=-1)
        veh_q_individual = self.project_vehicle(feat)        # (B, M, D)

        # Collaborative pass: self-attention among vehicle tokens, key-masked so
        # real vehicles never attend to padded ghost slots (m >= num_vehicle_i).
        # td["num_vehicle"] always has >= 1 real vehicle, so no all-masked row.
        q, k, v = self.veh_attn_qkv(veh_q_individual).chunk(3, dim=-1)
        heads = scaled_dot_product_attention(
            q, k, v, attn_mask=valid_veh.unsqueeze(1)        # (B,1,M) over keys
        )                                                    # (B, M, D)
        veh_q_collab = self.veh_attn_out(heads)              # (B, M, D)

        # Fuse individual + collaborative (RANS eq. 25 style): (B, M, D)
        return self.fuse(torch.cat([veh_q_individual, veh_q_collab], dim=-1))


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
        P: int = 3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.joint_mv_context = JointMVContext(embed_dim, P=P, linear_bias=linear_bias)
        self.dynamic_embedding = StaticEmbedding(embed_dim)

        self.pointer = PointerAttention(
            embed_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

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
