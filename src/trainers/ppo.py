import copy
from typing import Any, Union
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import BaseRL


class CriticNetwork(nn.Module):
    """Value network for PPO's advantage baseline (PPO-only; GRPO is critic-free)."""
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


class PPO(BaseRL):
    def __init__(
        self,
        env,
        policy,
        path_train_data: str,
        path_val_data: str,
        path_test_data: str,
        critic_kwargs: dict = {},
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # inner epoch, K
        batch_size: int = 1024,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        test_data_size: int = 1000,
        mini_batch_size: Union[int, float] = 0.25,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        log_on_step: bool = True,
        metrics: dict = {
            "train": ["reward", "loss", "surrogate_loss", "value_loss", "entropy"],
        },
        lr_scheduler_monitor: str = "val/reward",
        shuffle_train_dataloader: bool = True,
        dataloader_num_workers: int = 24,
        reload_train_dataloader: int = 4,
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
        self.save_hyperparameters()
        self.automatic_optimization = False  # PPO uses custom optimization routine
        self.critic = create_critic_from_actor(policy, **critic_kwargs)

        bs = self.data_cfg["batch_size"]
        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": bs // 4 if mini_batch_size >= bs else mini_batch_size,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "normalize_adv": normalize_adv,
            "max_grad_norm": max_grad_norm,
        }
        self.lr_scheduler_monitor = lr_scheduler_monitor

    def configure_optimizers(self):
        parameters = list(self.policy.parameters()) + list(self.critic.parameters())
        return torch.optim.AdamW(parameters, lr=1e-4)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Evaluate old actions, log probabilities, and rewards
        with torch.no_grad():
            td = self.env.reset(batch)  # note: clone needed for dataloader
            out = self.policy(td.clone(), self.env, phase=phase)

        if phase == "train":
            batch_size = out["actions"].shape[0]

            # infer batch size
            if isinstance(self.ppo_cfg["mini_batch_size"], float):
                mini_batch_size = int(batch_size * self.ppo_cfg["mini_batch_size"])
            elif isinstance(self.ppo_cfg["mini_batch_size"], int):
                mini_batch_size = self.ppo_cfg["mini_batch_size"]
            else:
                raise ValueError("mini_batch_size must be an integer or a float.")

            if mini_batch_size > batch_size:
                mini_batch_size = batch_size

            # Todo: Add support for multi dimensional batches
            td.set("logprobs", out["log_likelihood"])
            td.set("reward", out["reward"])
            td.set("action", out["actions"])

            idxs = torch.cat([torch.randperm(td.size(0)) for _ in range(self.ppo_cfg["ppo_epochs"])])
            size_inner = td.size(0)*self.ppo_cfg["ppo_epochs"]
            for i in range(0, size_inner, self.ppo_cfg["mini_batch_size"]):
                id_sub = idxs[i:i+self.ppo_cfg["mini_batch_size"]]
                sub_td = td[id_sub]
                previous_reward = sub_td["reward"].view(-1, 1)
                out = self.policy(  # note: remember to clone to avoid in-place replacements!
                    sub_td.clone(),
                    actions=sub_td["action"],
                    env=self.env,
                    return_entropy=True,
                    calc_reward=False,
                    return_sum_log_likelihood=False,
                )
                ll, entropy = out["log_likelihood"], out["entropy"]

                # Compute the ratio of probabilities of new and old actions
                ratio = torch.exp(ll.sum(dim=-1) - sub_td["logprobs"]).view(
                    -1, 1
                )  # [batch, 1]

                # Compute the advantage
                value_pred = self.critic(sub_td)  # [batch, 1]
                adv = previous_reward - value_pred.detach()

                # Normalize advantage
                if self.ppo_cfg["normalize_adv"]:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Compute the surrogate loss
                surrogate_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(
                        ratio,
                        1 - self.ppo_cfg["clip_range"],
                        1 + self.ppo_cfg["clip_range"],
                    )
                    * adv,
                ).mean()

                # compute value function loss
                value_loss = F.huber_loss(value_pred, previous_reward)

                # compute total loss
                loss = (
                    surrogate_loss
                    + self.ppo_cfg["vf_lambda"] * value_loss
                    - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                )

                # perform manual optimization following the Lightning routine
                # https://lightning.ai/docs/pytorch/stable/common/optimization.html
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

            out.update(
                {
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": value_loss,
                    "entropy": entropy.mean(),
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
