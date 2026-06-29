"""GRPO — group-relative policy optimization with a centered lexicographic-rank
advantage. Critic-free: the centered group rank IS the advantage.

Memory-efficient update:
  1. Roll out the full B*K group inside `torch.no_grad()` — no activation graph,
     so the B*K forward costs almost no memory. Reward, summed log-prob and actions
     are stashed in `td`.
  2. Re-run the policy in MINI-BATCHES (chunks of `mini_batch_size`) with grad on,
     replaying stored actions. Gradient materializes for one slice at a time.
  3. Clipped surrogate with the lexicographic rank as the advantage:
         ratio     = exp(ll_new - ll_old)
         surrogate = -min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv).mean()
         loss      = surrogate - entropy_lambda * entropy.mean()

The env must run in reward_mode='vector' so the rollout reward is the (B*K,3)
T-vector that the rank consumes.
"""
import numpy as np
import torch

from utils.ops import batchify, unbatchify
from .base import BaseRL

# Lexicographic separation constant for the val/lex_best model-selection scalar
# (T_1 weight = _LEX_C^2, T_2 = _LEX_C, T_3 = 1). Must dominate the T_2/T_3 range.
_LEX_C = 1.0e3


def centered_lex_rank(T):
    """Centered lexicographic rank advantage for one group of K solutions.

    Args:
        T: (K, P) array of T-vectors (LOWER is better), P = priority classes.
           Ranked lexicographically: T[:,0] (T_1) primary, then T_2, ..., T_P.

    Returns:
        (K,) float32 advantage in [-1, +1], mean == 0.
        Best solution (lowest T lex) → +1, worst → -1.
        K == 1 → all zeros.
        TIES: identical T-vectors share the average of the ranks they span →
        equal advantage, no spurious gradient. Whole group tied → all zeros
        (zero gradient; the dynamic-filter branch below drops these groups).
    """
    T = np.asarray(T, dtype=np.float64)
    K = T.shape[0]
    if K < 2:
        return np.zeros(K, dtype=np.float32)
    # lexsort: last key is primary → T[:,0] (T_1) is primary, T_P is last key.
    order = np.lexsort(T[:, ::-1].T)       # ascending: best (lowest T_1) first
    pos = np.arange(K - 1, -1, -1, dtype=np.float64)   # best pos → K-1
    # Average-rank ties: runs of identical T-vectors share the mean position.
    Ts = T[order]
    rank_sorted = pos.copy()
    i = 0
    while i < K:
        j = i + 1
        while j < K and np.array_equal(Ts[j], Ts[i]):
            j += 1
        if j - i > 1:
            rank_sorted[i:j] = pos[i:j].mean()
        i = j
    rank = np.empty(K, dtype=np.float64)
    rank[order] = rank_sorted
    adv = (rank - (K - 1) / 2.0) / ((K - 1) / 2.0)    # → [-1, +1], mean == 0
    return adv.astype(np.float32)


def _centered_lex_rank_batch(T_np):
    """Batched version of centered_lex_rank.

    Args:
        T_np: (B, K, P) float64 numpy array (LOWER is better).

    Returns:
        (B, K) float32 advantage in [-1, +1], mean==0 per group.
    """
    B, K, P = T_np.shape
    if K < 2:
        return np.zeros((B, K), dtype=np.float32)

    # Step 1: lexsort each group via successive stable argsorts (last->first key).
    # order[b, i] = index in original group that belongs at sorted position i.
    order = np.tile(np.arange(K, dtype=np.intp), (B, 1))       # (B, K)
    Bidx = np.arange(B, dtype=np.intp)[:, None]                 # (B, 1) for fancy index
    for p in range(P - 1, -1, -1):                             # last key = least significant
        col = T_np[Bidx, order, p]                            # (B, K) current values
        sub = np.argsort(col, axis=1, kind='stable')
        order = order[Bidx, sub]

    # Step 2: positions K-1 (best) down to 0 (worst), replicated for B groups.
    pos = np.arange(K - 1, -1, -1, dtype=np.float64)          # (K,)
    rank_sorted = np.broadcast_to(pos, (B, K)).copy()          # (B, K) writeable

    # Step 3: average rank for ties. Ties = consecutive sorted rows equal on ALL P cols.
    Ts = T_np[Bidx, order, :]                                 # (B, K, P) sorted
    # eq[b, i] == True if sorted row i+1 equals row i (all P coords).
    eq = np.all(Ts[:, 1:, :] == Ts[:, :-1, :], axis=2)        # (B, K-1) bool

    # Process each group's runs. For groups with NO ties (common case), skip.
    has_tie = eq.any(axis=1)                                   # (B,) bool
    for b in np.nonzero(has_tie)[0]:
        i = 0
        while i < K:
            j = i + 1
            while j < K and eq[b, j - 1]:
                j += 1
            if j - i > 1:
                rank_sorted[b, i:j] = pos[i:j].mean()
            i = j

    # Step 4: scatter back to original indices and normalise.
    rank = np.empty((B, K), dtype=np.float64)
    rank[Bidx, order] = rank_sorted
    adv = (rank - (K - 1) / 2.0) / ((K - 1) / 2.0)
    return adv.astype(np.float32)


class GRPO(BaseRL):
    def __init__(
        self,
        env,
        policy,
        path_train_data: str,
        path_val_data: str,
        path_test_data: str,
        group_size: int = 8,
        lr: float = 1e-4,
        entropy_lambda: float = 0.0,
        max_grad_norm: float = 1.0,
        clip_range: float = 0.2,
        mini_batch_size: int = 256,
        batch_size: int = 1024,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        test_data_size: int = 1000,
        metrics: dict = None,
        log_on_step: bool = True,
        dataloader_num_workers: int = 24,
        reload_train_dataloader: int = 4,
        **kwargs,   # swallow any remaining caller-compat args
    ):
        super().__init__(
            env, policy, path_train_data, path_val_data, path_test_data,
            batch_size=batch_size, train_data_size=train_data_size,
            val_data_size=val_data_size, test_data_size=test_data_size,
            metrics=metrics, log_on_step=log_on_step,
            dataloader_num_workers=dataloader_num_workers,
            reload_train_dataloader=reload_train_dataloader,
        )
        self.save_hyperparameters(ignore=["env", "policy"])
        self.automatic_optimization = False  # one explicit backward + step

        assert group_size >= 2, "GRPO needs group_size K >= 2 to form a ranking baseline"
        self.group_size = group_size
        self.lr = lr
        self.entropy_lambda = entropy_lambda
        self.max_grad_norm = max_grad_norm
        self.clip_range = clip_range
        self.mini_batch_size = mini_batch_size

    def configure_optimizers(self):
        # Policy params only — GRPO is critic-free.
        return torch.optim.AdamW(self.policy.parameters(), lr=self.lr)

    def _group_advantage(self, T_bk3):
        """(B, K, 3) T-vectors -> (B, K) centered lexicographic-rank advantage."""
        T = T_bk3.detach().cpu().numpy().astype(np.float64)
        return torch.from_numpy(_centered_lex_rank_batch(T))

    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        if phase == "train":
            return self._train_step(batch)
        return self._eval_step(batch, phase, dataloader_idx)

    def _train_step(self, batch):
        K = self.group_size

        # (1) Roll out the full B*K group WITHOUT a gradient graph. batchify each
        # instance K times (POMO-style, k*B+b layout), sample one rollout per copy.
        # No activations are retained, so the big B*K forward is cheap in memory.
        with torch.no_grad():
            td0 = self.env.reset(batch)
            B = td0.batch_size[0]
            # Encode B instances once. Encoder uses InstanceNorm (batch-independent) so
            # K copies of the same instance produce identical hidden states — no need to
            # encode B*K times. Batchify the shared hidden to match k*B+b layout of td.
            hidden0, _ = self.policy.encoder(td0)         # (B, N, embed_dim)
            td = batchify(td0, K)                         # (B*K, ...): k*B+b layout
            hidden = batchify(hidden0, K)                 # (B*K, N, embed_dim)
            out = self.policy(td.clone(), self.env, phase="train", hidden=hidden)
        reward = out["reward"]                        # (B*K, 3) T-vector (no grad)

        T = unbatchify(reward, K)                     # (B, K, 3)
        adv_bk = self._group_advantage(T)             # (B, K), detached
        # Re-flatten (B,K)->(B*K,) matching td's k*B+b layout (transpose then reshape).
        advantage = adv_bk.permute(1, 0).reshape(B * K).to(reward.device)

        # Stash the rollout's old log-prob, advantage, actions and reward on td so
        # the inner loop can replay them in mini-batches. reward must be present so
        # policy.forward can echo td["reward"] when calc_reward=False.
        td.set("reward", reward)                      # (B*K, P) T-vector
        td.set("logprobs", out["log_likelihood"])     # (B*K,) old sum log-prob
        td.set("advantage", advantage)                # (B*K,) centered rank
        td.set("action", out["actions"])

        # DAPO dynamic group filtering: a group whose K rollouts are all tied gives
        # an all-zero advantage (zero gradient). Drop those groups so compute is
        # spent only on groups that produce a signal; log the skipped fraction.
        active_mask = adv_bk.abs().amax(dim=1) > 1e-6     # (B,) bool
        skipped_frac = float((B - int(active_mask.sum())) / B)

        # Per-objective means for the learning-curve log; P = #priority classes
        # (inferred from the T-vector width — no hardcoded 3).
        P = T.shape[-1]
        Tk = [T[..., k].mean() for k in range(P)]

        # Whole batch inactive (every group tied): nothing to backward. Still log.
        if not bool(active_mask.any()):
            out_log = {"loss": torch.tensor(0.0), "reward": -Tk[0].reshape(1),
                       "skipped_frac": skipped_frac}
            out_log.update({f"T{k+1}_mean": Tk[k] for k in range(P)})
            metrics = self.log_metrics(out_log, "train")
            metrics["train/skipped_frac"] = skipped_frac
            return {"loss": 0.0, **metrics}

        # Keep only active groups: build (B*K,) active row indices in k*B+b order
        # (rows k*B+b for every active b) and slice td down to the active rollouts.
        active_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
        keep = (torch.arange(K).view(K, 1) * B + active_idx.view(1, -1)).reshape(-1)
        td = td[keep.to(reward.device)]

        # (2) Grad-enabled inner loop: one shuffled pass over active rollouts.
        mini_batch_size = max(1, min(self.mini_batch_size, td.size(0)))
        idxs = torch.randperm(td.size(0))

        opt = self.optimizers()
        loss = entropy = None
        for i in range(0, td.size(0), mini_batch_size):
            sub_td = td[idxs[i:i + mini_batch_size]]
            adv = sub_td["advantage"].view(-1, 1)

            out_i = self.policy(
                sub_td.clone(),
                actions=sub_td["action"],
                env=self.env,
                return_entropy=(self.entropy_lambda > 0),
                calc_reward=False,
                return_sum_log_likelihood=False,
            )
            ll = out_i["log_likelihood"]
            entropy = out_i.get("entropy", None)
            ratio = torch.exp(ll.sum(dim=-1) - sub_td["logprobs"]).view(-1, 1)

            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            surr = -torch.min(ratio * adv, clipped_ratio * adv).mean()
            loss = surr - self.entropy_lambda * entropy.mean() if entropy is not None else surr

            opt.zero_grad()
            self.manual_backward(loss)
            if self.max_grad_norm is not None:
                self.clip_gradients(opt, gradient_clip_val=self.max_grad_norm,
                                    gradient_clip_algorithm="norm")
            opt.step()

        out_log = {"loss": loss, "reward": -Tk[0].reshape(1),
                   "entropy": entropy.mean() if entropy is not None else torch.tensor(0.0),
                   "skipped_frac": skipped_frac}
        out_log.update({f"T{k+1}_mean": Tk[k] for k in range(P)})
        metrics = self.log_metrics(out_log, "train")
        metrics["train/skipped_frac"] = skipped_frac
        return {"loss": loss, **metrics}

    def _eval_step(self, batch, phase, dataloader_idx):
        # Deterministic greedy rollout (val/test decode type) on the B instances.
        with torch.no_grad():
            td0 = self.env.reset(batch)
            out = self.policy(td0.clone(), self.env, phase=phase)
        reward = out["reward"]                        # (B, P) T-vector
        T = reward.detach().cpu().numpy()
        B, P = T.shape[0], T.shape[1]
        for j in range(P):
            self.log(f"{phase}/T{j+1}_best", float(T[:, j].mean()),
                     on_epoch=True, sync_dist=True, batch_size=B)
        # Lexicographic model-selection scalar (T_1 >> T_2 >> ... >> T_P); higher is
        # better so ModelCheckpoint(mode='max') picks the best-lexicographic epoch.
        # Horner over the P columns (P=3 -> (T0*C+T1)*C+T2).
        lex = T[:, 0].copy()
        for j in range(1, P):
            lex = lex * _LEX_C + T[:, j]
        self.log(f"{phase}/lex_best", float(-lex.mean()),
                 on_epoch=True, sync_dist=True, batch_size=B)
        metrics = self.log_metrics({"reward": -reward[:, 0:1]}, phase, dataloader_idx)
        return {"loss": None, **metrics}
