"""D2 Phase 3: GRPO — group-relative policy optimization with a centered
lexicographic rank advantage (critic-free).

`class GRPO(PPO)` inherits everything (dataloaders, setup, log_metrics, the
mini-batch surrogate/clip) and overrides only the three pieces that differ from
weighted-sum PPO:

  * __init__            : adds `group_size` K (samples per instance).
  * configure_optimizers: optimizes ONLY policy params (the inherited critic is
                          excluded from both the optimizer and the loss).
  * shared_step         : batchify each instance to K, roll out, unbatchify the
                          (B*K, 3) T-vector to (B, K, 3), lexsort the K by
                          (T_1, T_2, T_3), advantage = centered rank (best -> +1,
                          worst -> -1, group mean == 0 baseline), then reuse PPO's
                          clipped surrogate with adv = rank and NO value loss.

rl/ppo.py is left UNCHANGED, so `--algo ppo` is byte-identical for A/B + rollback.
GRPO requires the env in reward_mode='vector' (Phase 2) so out["reward"] is (B*K, 3).
"""
import numpy as np
import torch

from rl.ppo import PPO
from common.ops import batchify, unbatchify


def centered_lex_rank(T):
    """Centered lexicographic rank advantage for one group of K solutions.

    Args:
        T: (K, 3) array of T-vectors (T positive, LOWER is better). Ranked
           lexicographically by (T_1, then T_2, then T_3).

    Returns:
        (K,) float32 advantage in [-1, 1]: the BEST solution (lowest T lex) gets
        +1, the worst -1, mean == 0 (group mean is the baseline). K==1 -> all 0.
    """
    T = np.asarray(T, dtype=np.float64)
    K = T.shape[0]
    if K < 2:
        return np.zeros(K, dtype=np.float32)
    # lexsort uses the LAST key as primary -> T[:,0] (T_1) primary, then T_2, T_3.
    order = np.lexsort((T[:, 2], T[:, 1], T[:, 0]))   # indices best -> worst
    rank = np.empty(K, dtype=np.float64)
    # best (order[0]) gets the highest rank K-1, worst gets 0.
    rank[order] = np.arange(K - 1, -1, -1)
    adv = (rank - (K - 1) / 2.0) / ((K - 1) / 2.0)    # [-1, 1], mean == 0
    return adv.astype(np.float32)


class GRPO(PPO):
    def __init__(self, *args, group_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        assert group_size >= 2, "GRPO needs group_size K >= 2 to form a ranking baseline"
        self.group_size = group_size
        # The inherited critic still exists (so super().__init__ doesn't break),
        # but it is excluded from the optimizer (configure_optimizers) and never
        # called in shared_step -> it receives NO gradient.

    def configure_optimizers(self):
        # Policy params only: GRPO has no value term, so the critic is not trained.
        return torch.optim.AdamW(self.policy.parameters(), lr=1e-4)

    def _group_advantage(self, T_bk3):
        """(B, K, 3) T-vectors -> (B, K) centered lexicographic-rank advantage."""
        B, K = T_bk3.shape[0], T_bk3.shape[1]
        T = T_bk3.detach().cpu().numpy()
        adv = np.empty((B, K), dtype=np.float32)
        for b in range(B):
            adv[b] = centered_lex_rank(T[b])
        return torch.from_numpy(adv)

    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        K = self.group_size

        # (a) roll out the group: batchify each instance K times (POMO-style).
        with torch.no_grad():
            td0 = self.env.reset(batch)
            td = batchify(td0, K)                      # (B*K, ...): instance-major
            out = self.policy(td.clone(), self.env, phase=phase)
        # out["reward"] is (B*K, 3) because env.reward_mode == 'vector' (Phase 2).

        if phase != "train":
            # Eval/val: scalarize the (B*K,3) reward to -T_1 ONLY for the monitor
            # metric (val/reward, mode='max'); this never touches gradients.
            reward = out["reward"]
            if reward.dim() == 2 and reward.shape[-1] == 3:
                out["reward"] = -reward[:, 0:1]
            metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
            return {"loss": out.get("loss", None), **metrics}

        B = td0.batch_size[0]
        reward = out["reward"]                          # (B*K, 3) T-vector
        T = unbatchify(reward, K)                       # (B, K, 3) — correct axis, no hand reshape

        adv_bk = self._group_advantage(T)               # (B, K), adv_bk[b,k] for instance b sample k
        # Flatten back to the (B*K,) layout that matches `td`. batchify produces a
        # K-MAJOR layout (flat index = k*B + b), and unbatchify inverts it to (B,K);
        # so to re-flatten we transpose (B,K)->(K,B) then row-major reshape -> k*B+b.
        advantage = adv_bk.permute(1, 0).reshape(B * K, 1).to(reward.device)  # (B*K, 1)

        # Per-objective means for the learning-curve sanity log (Phase 4).
        T1_mean = T[..., 0].mean()
        T2_mean = T[..., 1].mean()
        T3_mean = T[..., 2].mean()

        td.set("reward", reward)          # carried so policy.forward can echo it in the inner loop
        td.set("logprobs", out["log_likelihood"])
        td.set("advantage", advantage)
        td.set("action", out["actions"])

        batch_size = out["actions"].shape[0]            # B*K
        if isinstance(self.ppo_cfg["mini_batch_size"], float):
            mini_batch_size = int(batch_size * self.ppo_cfg["mini_batch_size"])
        else:
            mini_batch_size = self.ppo_cfg["mini_batch_size"]
        if mini_batch_size > batch_size:
            mini_batch_size = batch_size
        mini_batch_size = max(1, mini_batch_size)

        idxs = torch.cat([torch.randperm(td.size(0))
                          for _ in range(self.ppo_cfg["ppo_epochs"])])
        size_inner = td.size(0) * self.ppo_cfg["ppo_epochs"]
        loss = surrogate_loss = entropy = None
        for i in range(0, size_inner, mini_batch_size):
            id_sub = idxs[i:i + mini_batch_size]
            sub_td = td[id_sub]
            adv = sub_td["advantage"].view(-1, 1)       # rank (already centered) — NO critic
            out_i = self.policy(
                sub_td.clone(),
                actions=sub_td["action"],
                env=self.env,
                return_entropy=True,
                calc_reward=False,
                return_sum_log_likelihood=False,
            )
            ll, entropy = out_i["log_likelihood"], out_i["entropy"]
            ratio = torch.exp(ll.sum(dim=-1) - sub_td["logprobs"]).view(-1, 1)

            if self.ppo_cfg["normalize_adv"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surrogate_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio,
                            1 - self.ppo_cfg["clip_range"],
                            1 + self.ppo_cfg["clip_range"]) * adv,
            ).mean()

            # No value loss: group mean is the baseline (critic-free).
            loss = surrogate_loss - self.ppo_cfg["entropy_lambda"] * entropy.mean()

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            if self.ppo_cfg["max_grad_norm"] is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                    gradient_clip_algorithm="norm",
                )
            opt.step()

        out.update({
            "reward": -T1_mean.reshape(1),    # scalarized (-T_1) for monitor/ckpt only
            "loss": loss,
            "surrogate_loss": surrogate_loss,
            "entropy": entropy.mean(),
            "T1_mean": T1_mean,
            "T2_mean": T2_mean,
            "T3_mean": T3_mean,
        })

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
