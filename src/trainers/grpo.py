"""GRPO — group-relative policy optimization with a centered lexicographic-rank
advantage. Critic-free: the centered group rank IS the advantage.

Memory-efficient update:
  1. Roll out the full B*K group inside `torch.no_grad()` — no activation graph,
     so the B*K forward costs almost no memory. Reward, summed log-prob and actions
     are stashed in `td`.
  2. Re-run the policy in MINI-BATCHES (chunks of `mini_batch_size`) with grad on,
     replaying stored actions. Gradient materializes for one slice at a time.
  3. REINFORCE surrogate with the hierarchical rank as the advantage (a single
     opt.step after all chunks keeps theta fixed, so a PPO ratio would be == 1):
         variant P: per-step causal phase-masked advantage — a step taken in
             phase p gets sum_{q>=p} w_q*r_q renormalized over q>=p
             (centered_hier_rank_by_phase), so steps can only be credited for
             priority levels they could still influence.
         variant U: one combined scalar advantage per rollout, all steps.
         loss = surrogate - entropy_lambda * entropy.mean()

The env must run in reward_mode='vector' so the rollout reward is the (B*K,3)
T-vector that the rank consumes.
"""

import numpy as np
import torch

from utils.ops import batchify, unbatchify
from .base import BaseRL

# Lexicographic separation constant for the val/lex_best model-selection scalar
# (T_1 weight = _LEX_C^2, T_2 = _LEX_C, T_3 = 1). Must dominate the T_2/T_3 range.
_LEX_C = 1.0e4


def _centered_midrank(x):
    """x: (N, K) float32 — returns centered midrank in [-1, +1]; lower value = higher rank.
    Exactly zero-mean per row (midrank tie-averaging preserves the rank sum).
    """
    N, K = x.shape
    srt, order = x.sort(dim=1)  # ascending
    pos = torch.arange(K, device=x.device, dtype=x.dtype).expand(N, K)
    change = torch.ones(N, K, dtype=torch.bool, device=x.device)
    change[:, 1:] = srt[:, 1:] != srt[:, :-1]
    run_id = change.cumsum(dim=1) - 1  # tie-run index
    run_sum = torch.zeros(N, K, dtype=x.dtype, device=x.device)
    run_cnt = torch.zeros(N, K, dtype=x.dtype, device=x.device)
    run_sum.scatter_add_(1, run_id, pos)
    run_cnt.scatter_add_(1, run_id, torch.ones_like(pos))
    mid_sorted = (run_sum / run_cnt.clamp(min=1)).gather(1, run_id)
    rank = torch.empty_like(mid_sorted)
    rank.scatter_(1, order, mid_sorted)  # 0=best .. K-1=worst
    return 1.0 - 2.0 * rank / (K - 1)  # +1=best, -1=worst


def centered_hier_rank_by_phase(T, gamma=0.5):
    """Phase-conditional hierarchical rank advantage, fully on GPU.

    T : (B, K, P) float32 — T-vectors, LOWER is better.
    Returns (B, P, K): row p-1 is the advantage for a decode step taken in phase p
    (1-indexed class), i.e. sum_{q>=p} w_q * r_q renormalized over q>=p only —
    levels the step can no longer influence (q < p) are excluded.
    Row 0 (phase 1, all levels) is exactly the classic combined advantage.
    r_q = centered midrank of T[...,q] within the group, w_q = gamma^(q-1).
    """
    B, K, P = T.shape
    if K < 2:
        return torch.zeros(B, P, K, device=T.device, dtype=T.dtype)
    # Compute midrank for each level independently: reshape (B,K,P) -> (B*P, K)
    r = _centered_midrank(T.permute(0, 2, 1).reshape(B * P, K)).view(B, P, K)
    w = gamma ** torch.arange(P, device=T.device, dtype=T.dtype)
    lvl = torch.arange(P, device=T.device)
    rows = []
    for p in range(P):
        wp = torch.where(lvl >= p, w, torch.zeros_like(w))
        rows.append(torch.einsum("p,bpk->bk", wp / wp.sum(), r))
    return torch.stack(rows, dim=1)


def centered_hier_rank_gpu(T, gamma=0.5):
    """Hierarchical soft-lexicographic rank advantage, fully on GPU.

    T : (B, K, P) float32 — T-vectors, LOWER is better.
    Returns (B, K) float32 in [-1, +1], exactly zero-mean per group.

    A_i = sum_p gamma^(p-1) * r_{p,i} / sum_p gamma^(p-1)
    where r_p = centered midrank of T[...,p] within the group (independent per level).
    Adjacent priority levels trade off at a fixed 1/gamma gradient ratio, K-independent.
    Scales to any P without re-tuning: gradient share of level p = gamma^(p-1)/sum(gamma^q).
    """
    return centered_hier_rank_by_phase(T, gamma=gamma)[:, 0, :]


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
        adv_gamma: float = 0.5,
        metrics: dict = None,
        log_on_step: bool = True,
        train_buckets: list = None,
    ):
        super().__init__(
            env,
            policy,
            path_train_data,
            path_val_data,
            path_test_data,
            batch_size=batch_size,
            metrics=metrics,
            log_on_step=log_on_step,
            train_buckets=train_buckets,
        )
        self.save_hyperparameters(ignore=["env", "policy"])
        self.automatic_optimization = False  # one explicit backward + step

        assert group_size >= 2, (
            "GRPO needs group_size K >= 2 to form a ranking baseline"
        )
        self.group_size = group_size
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.max_grad_norm = max_grad_norm
        self.entropy_lambda = entropy_lambda
        self.mini_batch_size = mini_batch_size
        self.adv_gamma = adv_gamma

    def configure_optimizers(self):
        import math
        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

        opt = torch.optim.AdamW(self.policy.parameters(), lr=self.lr)
        warmup = LinearLR(
            opt,
            start_factor=self.lr_min / self.lr,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        T, r = self.cosine_steps, self.lr_min / self.lr
        # One-shot cosine that clamps at eta_min after T steps (no oscillation).
        cosine = LambdaLR(
            opt,
            lr_lambda=lambda s: max(
                r, r + 0.5 * (1 - r) * (1 + math.cos(math.pi * min(s, T) / T))
            ),
        )
        self._scheduler = SequentialLR(
            opt, [warmup, cosine], milestones=[self.warmup_steps]
        )
        return opt

    def _group_advantage(self, T_bk3):
        """(B, K, P) T-vectors on GPU -> (B, K) hierarchical rank advantage on GPU."""
        return centered_hier_rank_gpu(T_bk3.detach(), gamma=self.adv_gamma)

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
            hidden0, _ = self.policy.encoder(td0)  # (B, N, embed_dim)
            td = batchify(td0, K)  # (B*K, ...): k*B+b layout
            hidden = batchify(hidden0, K)  # (B*K, N, embed_dim)
            out = self.policy(td.clone(), self.env, phase="train", hidden=hidden)
        reward = out["reward"]  # (B*K, 3) T-vector (no grad)

        T = unbatchify(reward, K)  # (B, K, 3)
        adv_bk = self._group_advantage(T)  # (B, K), detached
        # Re-flatten (B,K)->(B*K,) matching td's k*B+b layout (transpose then reshape).
        advantage = adv_bk.permute(1, 0).reshape(B * K).to(reward.device)

        # Stash advantage, actions and reward on td for mini-batch replay.
        td.set("reward", reward)  # (B*K, P) T-vector
        td.set("advantage", advantage)  # (B*K,) centered rank
        td.set("action", out["actions"])
        if self.env.variant == "P":
            # Causal phase-masked advantage: (B,P,K) per-phase rank combinations,
            # re-laid-out to td's k*B+b rows -> (B*K, P). A decode step taken in
            # phase p gathers column p-1 (levels q<p are causally out of reach).
            adv_phase = centered_hier_rank_by_phase(T.detach(), gamma=self.adv_gamma)
            td.set("adv_phase", adv_phase.permute(2, 0, 1).reshape(B * K, -1))

        # DAPO dynamic group filtering: a group whose K rollouts are all tied gives
        # an all-zero advantage (zero gradient). Drop those groups so compute is
        # spent only on groups that produce a signal; log the skipped fraction.
        active_mask = adv_bk.abs().amax(dim=1) > 1e-6  # (B,) bool, on GPU
        skipped_frac = float((B - int(active_mask.sum())) / B)

        # Per-objective means for the learning-curve log; P = #priority classes
        # (inferred from the T-vector width — no hardcoded 3).
        P = T.shape[-1]
        Tk = [T[..., k].mean() for k in range(P)]

        # Whole batch inactive (every group tied): nothing to backward. Still log.
        if not bool(active_mask.any()):
            try:
                self.log(
                    "train/skipped_frac",
                    skipped_frac,
                    on_step=True,
                    prog_bar=False,
                    batch_size=B,
                    sync_dist=True,
                )
            except Exception:
                pass
            out_log = {"loss": torch.tensor(0.0), "reward": -Tk[0].reshape(1)}
            out_log.update({f"T{k + 1}_mean": Tk[k] for k in range(P)})
            metrics = self.log_metrics(out_log, "train")
            return {"loss": 0.0, "train/skipped_frac": skipped_frac, **metrics}

        # Keep only active groups: build (B_a*K,) active row indices in k*B_a+b' order.
        dev = reward.device
        active_idx = active_mask.nonzero(as_tuple=False).flatten()  # (B_a,) GPU
        keep = (
            torch.arange(K, device=dev).view(K, 1) * B + active_idx.view(1, -1)
        ).reshape(-1)
        td = td[keep]  # (B_a*K, ...) k*B_a+b' layout
        B_a = len(active_idx)

        # (2) Grad-enabled inner loop grouped by instance.
        # Each mini-batch = mb_inst unique instances × K rollouts = same samples/step
        # as before, but the encoder runs only on mb_inst rows (not mb_inst*K rows).
        # Total encoder work: B_a (≈ B) samples vs B_a*K = B*K in the old flat loop.
        mb_inst = max(1, self.mini_batch_size // K)
        inst_perm = torch.randperm(B_a, device=dev)  # shuffle on GPU

        # (2) Gradient accumulation: single optimizer step over all mini-batches.
        # zero_grad once before the loop; each chunk accumulates grads; step once
        # after. Policy stays at θ₀ for all ll evaluations → fully on-policy.
        opt = self.optimizers()
        opt.zero_grad()
        loss_acc = torch.tensor(0.0, device=dev)
        entropy_acc = 0.0

        for j in range(0, B_a, mb_inst):
            inst_idx = inst_perm[j : j + mb_inst]  # (actual_mb,) GPU
            orig_inst_idx = active_idx[inst_idx]  # (actual_mb,) GPU; B-indices

            # Collect all K rollouts of these instances: k*actual_mb+i layout
            actual_mb = len(inst_idx)
            weight = actual_mb / B_a  # proportional chunk weight
            flat_row_idx = (
                torch.arange(K, device=dev).view(K, 1) * B_a + inst_idx.view(1, -1)
            ).reshape(-1)  # (actual_mb*K,)
            sub_td = td[flat_row_idx]

            # Encode only the actual_mb unique instances (with grad).
            sub_hidden, _ = self.policy.encoder(
                td0[orig_inst_idx.to(reward.device)]
            )  # (actual_mb, N, D)
            sub_hidden_bk = batchify(sub_hidden, K)  # (actual_mb*K, N, D)

            out_i = self.policy(
                sub_td.clone(),
                actions=sub_td["action"],
                env=self.env,
                hidden=sub_hidden_bk,
                calc_reward=False,
                return_sum_log_likelihood=False,
                return_entropy=True,
            )
            ll = out_i["log_likelihood"]  # (actual_mb*K, dec_steps), junk=0
            entropy = out_i["entropy"]
            vm = out_i.get("valid_mask")  # (actual_mb*K, dec_steps) or None

            # Length-normalize: sum over valid steps / valid count (excludes post-done junk).
            if self.env.variant == "P":
                # Per-step causal advantage: gather the phase-conditional rank each
                # decision was taken in. Padding steps have ll==0 and a clamped
                # (finite) phase, so they contribute exactly 0.
                phase_idx = (out_i["phase"].long() - 1).clamp(
                    0, sub_td["adv_phase"].shape[-1] - 1
                )
                adv_step = sub_td["adv_phase"].gather(1, phase_idx)  # (mb*K, dec_steps)
                if vm is not None:
                    valid_len = vm.float().sum(dim=-1).clamp(min=1)
                else:
                    valid_len = float(ll.shape[-1])
                surr = -((ll * adv_step).sum(dim=-1) / valid_len).mean()
            else:
                adv = sub_td["advantage"].view(-1, 1)
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
            self.clip_gradients(
                opt,
                gradient_clip_val=self.max_grad_norm,
                gradient_clip_algorithm="norm",
            )
        opt.step()
        self._scheduler.step()

        loss_log = loss_acc
        try:
            self.log(
                "train/skipped_frac",
                skipped_frac,
                on_step=True,
                prog_bar=False,
                batch_size=B,
                sync_dist=True,
            )
            self.log(
                "train/entropy",
                entropy_acc,
                on_step=True,
                prog_bar=True,
                batch_size=B,
                sync_dist=True,
            )
            lr_now = self._scheduler.optimizer.param_groups[0]["lr"]
            self.log(
                "train/lr",
                lr_now,
                on_step=True,
                prog_bar=False,
                batch_size=B,
                sync_dist=True,
            )
        except Exception:
            pass
        out_log = {"loss": loss_log, "reward": -Tk[0].reshape(1)}
        out_log.update({f"T{k + 1}_mean": Tk[k] for k in range(P)})
        metrics = self.log_metrics(out_log, "train")
        return {"loss": loss_log, "train/skipped_frac": skipped_frac, **metrics}

    def _eval_step(self, batch, phase, dataloader_idx):
        # Deterministic greedy rollout (val/test decode type) on the B instances.
        with torch.no_grad():
            td0 = self.env.reset(batch)
            out = self.policy(td0.clone(), self.env, phase=phase)
        reward = out["reward"]  # (B, P) T-vector, on GPU
        B, P = reward.shape
        for j in range(P):
            self.log(
                f"{phase}/T{j + 1}_best",
                float(reward[:, j].mean()),
                on_epoch=True,
                sync_dist=True,
                batch_size=B,
            )
        # Lex scalar on GPU via Horner: (T0*C+T1)*C+T2
        lex = reward[:, 0].clone()
        for j in range(1, P):
            lex = lex * _LEX_C + reward[:, j]
        self.log(
            f"{phase}/lex_best",
            float(-lex.mean()),
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(f"{phase}/lex_best", float(-lex.mean()),
                 on_epoch=True, sync_dist=True, batch_size=B)
        metrics = self.log_metrics({"reward": -reward[:, 0:1]}, phase, dataloader_idx)
        return {"loss": None, **metrics}
