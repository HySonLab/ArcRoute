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


def centered_lex_rank(T):
    """CPU reference: centered lex-rank advantage for ONE group of K solutions.

    T   : (K, P) array — T-vectors, LOWER is better.
    Returns (K,) float32 in [-1, +1], mean==0.
    Used as ground-truth in unit tests; training uses centered_lex_rank_gpu.
    """
    T = np.asarray(T, dtype=np.float64)
    K = T.shape[0]
    if K < 2:
        return np.zeros(K, dtype=np.float32)
    order = np.lexsort(T[:, ::-1].T)
    pos = np.arange(K - 1, -1, -1, dtype=np.float64)
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
    return ((rank - (K - 1) / 2.0) / ((K - 1) / 2.0)).astype(np.float32)

# Lexicographic separation constant for the val/lex_best model-selection scalar
# (T_1 weight = _LEX_C^2, T_2 = _LEX_C, T_3 = 1). Must dominate the T_2/T_3 range.
_LEX_C = 1.0e3


def centered_lex_rank_gpu(T):
    """Centered lexicographic rank advantage entirely on GPU.

    T   : (B, K, P) float32 tensor — T-vectors, LOWER is better.
    Returns (B, K) float32 advantage in [-1, +1], mean==0 per group.

    Lexsort via P successive stable argsorts (least-significant column first).
    Tie averaging: runs of identical sorted rows share the mean position.
    Rare tied groups fall back to a tiny CPU loop over only those rows.
    """
    B, K, P = T.shape
    dev = T.device
    if K < 2:
        return torch.zeros(B, K, device=dev)

    # --- lexsort: P stable argsorts, least-significant first ---
    order = torch.arange(K, device=dev).unsqueeze(0).expand(B, -1).clone()  # (B, K)
    idx3 = order.unsqueeze(-1).expand(B, K, P)
    for p in range(P - 1, -1, -1):
        col = T.gather(1, idx3)[:, :, p]                    # (B, K) current column
        order = order.gather(1, col.argsort(dim=1, stable=True))
        idx3 = order.unsqueeze(-1).expand(B, K, P)

    # --- positions: best (rank 0) → K-1, worst → 0 ---
    pos = torch.arange(K - 1, -1, -1, dtype=T.dtype, device=dev)  # (K,)
    rank_sorted = pos.unsqueeze(0).expand(B, K).clone()             # (B, K)

    # --- tie averaging ---
    Ts = T.gather(1, idx3)                                  # (B, K, P) sorted
    eq = (Ts[:, 1:] == Ts[:, :-1]).all(dim=2)              # (B, K-1) bool
    if eq.any():
        # Only process groups that actually have ties (rare path).
        has_tie = eq.any(dim=1)                             # (B,) bool
        tied_bs = has_tie.nonzero(as_tuple=True)[0].tolist()
        rs = rank_sorted.cpu().numpy()
        eq_np = eq.cpu().numpy()
        pos_np = pos.cpu().numpy()
        for b in tied_bs:
            i = 0
            while i < K:
                j = i + 1
                while j < K and eq_np[b, j - 1]:
                    j += 1
                if j - i > 1:
                    rs[b, i:j] = pos_np[i:j].mean()
                i = j
        rank_sorted = torch.from_numpy(rs).to(dev)

    # --- scatter back and normalise ---
    rank = torch.empty(B, K, dtype=T.dtype, device=dev)
    rank.scatter_(1, order, rank_sorted)
    return (rank - (K - 1) / 2.0) / ((K - 1) / 2.0)      # [-1, +1], mean==0



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
        lr_min: float = 1e-5,
        warmup_steps: int = 100,
        cosine_steps: int = 10000,
        max_grad_norm: float = 1.0,
        entropy_lambda: float = 0.02,
        mini_batch_size: int = 256,
        batch_size: int = 1024,
        metrics: dict = None,
        log_on_step: bool = True,
        train_buckets: list = None,
    ):
        super().__init__(
            env, policy, path_train_data, path_val_data, path_test_data,
            batch_size=batch_size, metrics=metrics, log_on_step=log_on_step,
            train_buckets=train_buckets,
        )
        self.save_hyperparameters(ignore=["env", "policy"])
        self.automatic_optimization = False  # one explicit backward + step

        assert group_size >= 2, "GRPO needs group_size K >= 2 to form a ranking baseline"
        self.group_size = group_size
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.max_grad_norm = max_grad_norm
        self.entropy_lambda = entropy_lambda
        self.mini_batch_size = mini_batch_size

    def configure_optimizers(self):
        import math
        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
        opt = torch.optim.AdamW(self.policy.parameters(), lr=self.lr)
        warmup = LinearLR(opt, start_factor=self.lr_min / self.lr, end_factor=1.0,
                          total_iters=self.warmup_steps)
        T, r = self.cosine_steps, self.lr_min / self.lr
        # One-shot cosine that clamps at eta_min after T steps (no oscillation).
        cosine = LambdaLR(opt, lr_lambda=lambda s: max(
            r, r + 0.5 * (1 - r) * (1 + math.cos(math.pi * min(s, T) / T))
        ))
        self._scheduler = SequentialLR(opt, [warmup, cosine],
                                       milestones=[self.warmup_steps])
        return opt

    def _group_advantage(self, T_bk3):
        """(B, K, P) T-vectors on GPU -> (B, K) centered lex-rank advantage on GPU."""
        return centered_lex_rank_gpu(T_bk3.detach())

    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        if phase == "train":
            return self._train_step(batch)
        return self._eval_step(batch, phase, dataloader_idx)

    def _train_step(self, batch):
        K = self.group_size

        # td0 lives outside no_grad so the inner loop can index it for per-group
        # encoder calls. env.reset is pure data preprocessing — no grad needed.
        td0 = self.env.reset(batch)
        B = td0.batch_size[0]

        # (1) Roll out the full B*K group WITHOUT a gradient graph. batchify each
        # instance K times (POMO-style, k*B+b layout), sample one rollout per copy.
        # No activations are retained, so the big B*K forward is cheap in memory.
        with torch.no_grad():
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

        # Stash advantage, actions and reward on td for mini-batch replay.
        td.set("reward", reward)                      # (B*K, P) T-vector
        td.set("advantage", advantage)                # (B*K,) centered rank
        td.set("action", out["actions"])

        # DAPO dynamic group filtering: a group whose K rollouts are all tied gives
        # an all-zero advantage (zero gradient). Drop those groups so compute is
        # spent only on groups that produce a signal; log the skipped fraction.
        active_mask = adv_bk.abs().amax(dim=1) > 1e-6     # (B,) bool, on GPU
        skipped_frac = float((B - int(active_mask.sum())) / B)

        # Per-objective means for the learning-curve log; P = #priority classes
        # (inferred from the T-vector width — no hardcoded 3).
        P = T.shape[-1]
        Tk = [T[..., k].mean() for k in range(P)]

        # Whole batch inactive (every group tied): nothing to backward. Still log.
        if not bool(active_mask.any()):
            try:
                self.log("train/skipped_frac", skipped_frac, on_step=True, prog_bar=False,
                         batch_size=B, sync_dist=True)
            except Exception:
                pass
            out_log = {"loss": torch.tensor(0.0), "reward": -Tk[0].reshape(1)}
            out_log.update({f"T{k+1}_mean": Tk[k] for k in range(P)})
            metrics = self.log_metrics(out_log, "train")
            return {"loss": 0.0, "train/skipped_frac": skipped_frac, **metrics}

        # Keep only active groups: build (B_a*K,) active row indices in k*B_a+b' order.
        dev = reward.device
        active_idx = active_mask.nonzero(as_tuple=False).flatten()          # (B_a,) GPU
        keep = (torch.arange(K, device=dev).view(K, 1) * B
                + active_idx.view(1, -1)).reshape(-1)
        td = td[keep]                      # (B_a*K, ...) k*B_a+b' layout
        B_a = len(active_idx)

        # (2) Grad-enabled inner loop grouped by instance.
        # Each mini-batch = mb_inst unique instances × K rollouts = same samples/step
        # as before, but the encoder runs only on mb_inst rows (not mb_inst*K rows).
        # Total encoder work: B_a (≈ B) samples vs B_a*K = B*K in the old flat loop.
        mb_inst = max(1, self.mini_batch_size // K)
        inst_perm = torch.randperm(B_a, device=dev)          # shuffle on GPU

        # (2) Gradient accumulation: single optimizer step over all mini-batches.
        # zero_grad once before the loop; each chunk accumulates grads; step once
        # after. Policy stays at θ₀ for all ll evaluations → fully on-policy.
        opt = self.optimizers()
        opt.zero_grad()
        loss_acc = torch.tensor(0.0, device=dev)
        entropy_acc = 0.0

        for j in range(0, B_a, mb_inst):
            inst_idx = inst_perm[j : j + mb_inst]            # (actual_mb,) GPU
            orig_inst_idx = active_idx[inst_idx]             # (actual_mb,) GPU; B-indices

            # Collect all K rollouts of these instances: k*actual_mb+i layout
            actual_mb = len(inst_idx)
            weight = actual_mb / B_a                         # proportional chunk weight
            flat_row_idx = (
                torch.arange(K, device=dev).view(K, 1) * B_a
                + inst_idx.view(1, -1)
            ).reshape(-1)                                     # (actual_mb*K,)
            sub_td = td[flat_row_idx]
            adv = sub_td["advantage"].view(-1, 1)

            # Encode only the actual_mb unique instances (with grad).
            sub_hidden, _ = self.policy.encoder(
                td0[orig_inst_idx.to(reward.device)]
            )                                                 # (actual_mb, N, D)
            sub_hidden_bk = batchify(sub_hidden, K)          # (actual_mb*K, N, D)

            out_i = self.policy(
                sub_td.clone(),
                actions=sub_td["action"],
                env=self.env,
                hidden=sub_hidden_bk,
                calc_reward=False,
                return_sum_log_likelihood=False,
                return_entropy=True,
            )
            ll = out_i["log_likelihood"]             # (actual_mb*K, dec_steps), junk=0
            entropy = out_i["entropy"]
            vm = out_i.get("valid_mask")             # (actual_mb*K, dec_steps) or None

            # Length-normalize: sum over valid steps / valid count (excludes post-done junk).
            if vm is not None:
                valid_len = vm.float().sum(dim=-1).clamp(min=1)
                ll_normed = ll.sum(dim=-1) / valid_len
            else:
                ll_normed = ll.mean(dim=-1)
            surr = -(ll_normed.view(-1, 1) * adv).mean()
            loss_i = surr - self.entropy_lambda * entropy.mean()

            # Accumulate scaled gradient (no opt.step here).
            self.manual_backward(loss_i * weight)
            loss_acc = loss_acc + loss_i.detach() * weight
            entropy_acc += float(entropy.mean().detach()) * weight

        # Single gradient step after full accumulation.
        if self.max_grad_norm is not None:
            self.clip_gradients(opt, gradient_clip_val=self.max_grad_norm,
                                gradient_clip_algorithm="norm")
        opt.step()
        self._scheduler.step()

        loss_log = loss_acc
        try:
            self.log("train/skipped_frac", skipped_frac, on_step=True, prog_bar=False,
                     batch_size=B, sync_dist=True)
            self.log("train/entropy", entropy_acc, on_step=True, prog_bar=True,
                     batch_size=B, sync_dist=True)
            lr_now = self._scheduler.optimizer.param_groups[0]["lr"]
            self.log("train/lr", lr_now, on_step=True, prog_bar=False,
                     batch_size=B, sync_dist=True)
        except Exception:
            pass
        out_log = {"loss": loss_log, "reward": -Tk[0].reshape(1)}
        out_log.update({f"T{k+1}_mean": Tk[k] for k in range(P)})
        metrics = self.log_metrics(out_log, "train")
        return {"loss": loss_log, "train/skipped_frac": skipped_frac, **metrics}

    def _eval_step(self, batch, phase, dataloader_idx):
        # Deterministic greedy rollout (val/test decode type) on the B instances.
        with torch.no_grad():
            td0 = self.env.reset(batch)
            out = self.policy(td0.clone(), self.env, phase=phase)
        reward = out["reward"]                        # (B, P) T-vector, on GPU
        B, P = reward.shape
        for j in range(P):
            self.log(f"{phase}/T{j+1}_best", float(reward[:, j].mean()),
                     on_epoch=True, sync_dist=True, batch_size=B)
        # Lex scalar on GPU via Horner: (T0*C+T1)*C+T2
        lex = reward[:, 0].clone()
        for j in range(1, P):
            lex = lex * _LEX_C + reward[:, j]
        self.log(f"{phase}/lex_best", float(-lex.mean()),
                 on_epoch=True, sync_dist=True, batch_size=B)
        metrics = self.log_metrics({"reward": -reward[:, 0:1]}, phase, dataloader_idx)
        return {"loss": None, **metrics}
