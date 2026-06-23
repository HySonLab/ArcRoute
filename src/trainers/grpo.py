"""GRPO — group-relative policy optimization with a centered lexicographic-rank
advantage. A STANDALONE sibling of PPO (both inherit `BaseRL`); GRPO does NOT
depend on PPO.

This is the rl4co/POMO-style design: vanilla REINFORCE with a group baseline, but
the baseline/advantage is a *lexicographic rank* over the K group samples instead
of the group mean (the project's multi-objective T₁≺T₂≺T₃ signal).

What it deliberately does NOT have (vs a PPO-style trainer):
  * no critic / value network,
  * no clipped surrogate ratio,
  * no inner ppo_epochs loop (log-probs from the rollout are used directly),
  * no mini_batch_size (one rollout -> one gradient step).
This makes it both simpler and much faster: a single grad-enabled rollout + one
backward, with the encoder ENCODED ONCE and shared across the K group (correct
here because there is a single forward/backward — no stale-weight issue).

Loss (REINFORCE with rank baseline):
    advantage = centered_lex_rank(T)        # (B,K), detached, in [-1,1], group-mean 0
    loss = -(advantage * sum_log_prob).mean()  [ - entropy_lambda * entropy ]

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
        T: (K, 3) array of T-vectors (T positive, LOWER is better). Ranked
           lexicographically by (T_1, then T_2, then T_3).

    Returns:
        (K,) float32 advantage in [-1, 1]: the BEST solution (lowest T lex) gets
        +1, the worst -1, mean == 0 (group mean is the baseline). K==1 -> all 0.

        TIES: solutions with an IDENTICAL T-vector are EQUALLY good, so they share
        the AVERAGE of the ranks they span -> equal advantage (no spurious gradient
        from the lexsort tiebreak). When the whole group is tied (common once T_1
        saturates) every advantage is 0 -> that group contributes no gradient, the
        DAPO "zero-variance masking" behaviour. With NO ties this is byte-identical
        to the plain integer-rank version.
    """
    T = np.asarray(T, dtype=np.float64)
    K = T.shape[0]
    if K < 2:
        return np.zeros(K, dtype=np.float32)
    # lexsort uses the LAST key as primary -> T[:,0] (T_1) primary, then T_2, T_3.
    order = np.lexsort((T[:, 2], T[:, 1], T[:, 0]))   # indices best -> worst
    # Positional rank: best position -> K-1, worst -> 0.
    pos = np.arange(K - 1, -1, -1, dtype=np.float64)
    # Average-rank ties: scan the sorted order, and for each run of identical
    # T-vectors replace their positional ranks with the run's mean.
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
    adv = (rank - (K - 1) / 2.0) / ((K - 1) / 2.0)    # [-1, 1], mean == 0
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
        batch_size: int = 1024,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        test_data_size: int = 1000,
        metrics: dict = None,
        log_on_step: bool = True,
        shuffle_train_dataloader: bool = True,
        dataloader_num_workers: int = 24,
        reload_train_dataloader: int = 4,
        **kwargs,   # swallow PPO-only args (mini_batch_size, clip_range, ...) for caller compat
    ):
        super().__init__(
            env, policy, path_train_data, path_val_data, path_test_data,
            batch_size=batch_size, train_data_size=train_data_size,
            val_data_size=val_data_size, test_data_size=test_data_size,
            metrics=metrics, log_on_step=log_on_step,
            shuffle_train_dataloader=shuffle_train_dataloader,
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

    def configure_optimizers(self):
        # Policy params only — GRPO is critic-free.
        return torch.optim.AdamW(self.policy.parameters(), lr=self.lr)

    def _group_advantage(self, T_bk3):
        """(B, K, 3) T-vectors -> (B, K) centered lexicographic-rank advantage."""
        B, K = T_bk3.shape[0], T_bk3.shape[1]
        T = T_bk3.detach().cpu().numpy()
        adv = np.empty((B, K), dtype=np.float32)
        for b in range(B):
            adv[b] = centered_lex_rank(T[b])
        return torch.from_numpy(adv)

    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        if phase == "train":
            return self._train_step(batch)
        return self._eval_step(batch, phase, dataloader_idx)

    def _train_step(self, batch):
        K = self.group_size
        td0 = self.env.reset(batch)
        B = td0.batch_size[0]

        # Encoder sharing: encode the B instances ONCE (with grad) and replicate the
        # output to the B*K group, so the encoder is not re-run per sample. Correct
        # here because there is a SINGLE forward/backward (no stale-weight issue).
        hidden0, _ = self.policy.encoder(td0)
        hidden = batchify(hidden0, K)                 # (B*K, num_nodes, embed_dim)
        td = batchify(td0, K)                         # (B*K, ...): k*B+b layout

        out = self.policy(td.clone(), self.env, phase="train",
                          hidden=hidden, return_entropy=True)
        reward = out["reward"]                        # (B*K, 3) T-vector (no grad)
        ll = out["log_likelihood"]                    # (B*K,) sum log-prob (grad)

        T = unbatchify(reward, K)                     # (B, K, 3)
        adv_bk = self._group_advantage(T)             # (B, K), detached
        # Re-flatten (B,K)->(B*K,) matching td's k*B+b layout (transpose then reshape).
        advantage = adv_bk.permute(1, 0).reshape(B * K).to(ll.device)

        entropy = out["entropy"].mean()
        # REINFORCE with the rank baseline: advantage is a constant weight on logp.
        loss = -(advantage * ll).mean() - self.entropy_lambda * entropy

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        if self.max_grad_norm is not None:
            self.clip_gradients(opt, gradient_clip_val=self.max_grad_norm,
                                gradient_clip_algorithm="norm")
        opt.step()

        T1, T2, T3 = T[..., 0].mean(), T[..., 1].mean(), T[..., 2].mean()
        out_log = {"loss": loss, "reward": -T1.reshape(1), "entropy": entropy,
                   "T1_mean": T1, "T2_mean": T2, "T3_mean": T3}
        metrics = self.log_metrics(out_log, "train")
        return {"loss": loss, **metrics}

    def _eval_step(self, batch, phase, dataloader_idx):
        # Deterministic greedy rollout (val/test decode type) on the B instances.
        with torch.no_grad():
            td0 = self.env.reset(batch)
            out = self.policy(td0.clone(), self.env, phase=phase)
        reward = out["reward"]                        # (B, 3) T-vector
        T = reward.detach().cpu().numpy()
        B = T.shape[0]
        for j, name in enumerate(("T1", "T2", "T3")):
            self.log(f"{phase}/{name}_best", float(T[:, j].mean()),
                     on_epoch=True, sync_dist=True, batch_size=B)
        # Lexicographic model-selection scalar (T_1 >> T_2 >> T_3); higher = better
        # so ModelCheckpoint(mode='max') picks the best-lexicographic epoch.
        lex = (T[:, 0] * _LEX_C + T[:, 1]) * _LEX_C + T[:, 2]
        self.log(f"{phase}/lex_best", float(-lex.mean()),
                 on_epoch=True, sync_dist=True, batch_size=B)
        metrics = self.log_metrics({"reward": -reward[:, 0:1]}, phase, dataloader_idx)
        return {"loss": None, **metrics}
