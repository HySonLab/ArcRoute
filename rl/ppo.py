from typing import Union
from torch import  nn
from typing import Any, Union, Iterable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import LightningModule
from functools import partial
from .critic import create_critic_from_actor

class PPO(LightningModule):
    def __init__(
        self,
        env,
        policy,
        critic_kwargs: dict = {},
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # inner epoch, K
        batch_size: int = 1024,
        train_data_size: int=100000,
        val_data_size: int=10000,
        test_data_size: int=1000,
        generate_default_data: bool = False,
        mini_batch_size: Union[int, float] = 0.25,  # 0.25,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        log_on_step: bool = True,
        metrics: dict = {
            "train": ["reward", "loss", "surrogate_loss", "value_loss", "entropy"],
        },
        optimizer: Union[str, torch.optim.Optimizer, partial] = "Adam",
        optimizer_kwargs: dict = {"lr": 1e-4},
        lr_scheduler: Union[str, torch.optim.lr_scheduler.LRScheduler, partial] = None,
        lr_scheduler_kwargs: dict = {
            "milestones": [80, 95],
            "gamma": 0.1,
        },
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str = "val/reward",
        shuffle_train_dataloader: bool = True,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # PPO uses custom optimization routine
        self.critic = create_critic_from_actor(policy, **critic_kwargs)
        self.env = env
        self.policy = policy
        self.mini_batch_size = mini_batch_size
        self.train_data_size = train_data_size

        self.instantiate_metrics(metrics)
        self.log_on_step = log_on_step


        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "normalize_adv": normalize_adv,
            "max_grad_norm": max_grad_norm,
        }

        self.data_cfg = {
            "batch_size": batch_size,
            "val_batch_size": batch_size,
            "test_batch_size": batch_size,
            "generate_default_data": generate_default_data,
            "train_data_size": train_data_size,
            "val_data_size": val_data_size,
            "test_data_size": test_data_size,
        }

        self.lr_scheduler_monitor=lr_scheduler_monitor
        self.shuffle_train_dataloader = shuffle_train_dataloader
        self.dataloader_num_workers = dataloader_num_workers

    def instantiate_metrics(self, metrics: dict):
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])
        self.log_on_step = metrics.get("log_on_step", True)
    
    def log_metrics(
        self, metric_dict: dict, phase: str, dataloader_idx: Union[int, None] = None
    ):
        """Log metrics to logger and progress bar"""
        metrics = getattr(self, f"{phase}_metrics")
        dataloader_name = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            dataloader_name = "/" + self.dataloader_names[dataloader_idx]
        metrics = {
            f"{phase}/{k}{dataloader_name}": v.mean()
            if isinstance(v, torch.Tensor)
            else v
            for k, v in metric_dict.items()
            if k in metrics
        }
        log_on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,  # we add manually above
        )
        return metrics

    def setup(self, stage="fit"):
        train_bs, val_bs, test_bs = (
            self.data_cfg["batch_size"],
            self.data_cfg["val_batch_size"],
            self.data_cfg["test_batch_size"],
        )
        self.train_batch_size = train_bs
        self.val_batch_size = train_bs if val_bs is None else val_bs
        self.test_batch_size = self.val_batch_size if test_bs is None else test_bs

        self.train_dataset = self.env.dataset(self.data_cfg["train_data_size"])
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"])
        self.test_dataset = self.env.dataset(self.data_cfg["test_data_size"])
        self.dataloader_names = None
        self.setup_loggers()

    def setup_loggers(self):
        """Log all hyperparameters except those in `nn.Module`"""
        if self.loggers is not None:
            hparams_save = {
                k: v for k, v in self.hparams.items() if not isinstance(v, nn.Module)
            }
            for logger in self.loggers:
                logger.log_hyperparams(hparams_save)
                logger.log_graph(self)
                logger.save()

    def configure_optimizers(self):
        parameters = list(self.policy.parameters()) + list(self.critic.parameters())

        optimizer = torch.optim.AdamW(
            parameters,
            lr=1e-4
        )
        return optimizer
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a MultiStepLR scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.MultiStepLR):
            sch.step()

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        
        with torch.no_grad():
            td = self.env.reset(batch)  # note: clone needed for dataloader
            out = self.policy(td.clone(), self.env, 
                              phase=phase, calc_reward=True, return_sum_log_likelihood=True)

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

            # Inherit the dataset class from the environment for efficiency
            dataset = self.env.dataset_cls(td)
            dataloader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
            )

            for _ in range(self.ppo_cfg["ppo_epochs"]):  # PPO inner epoch, K
                for sub_td in dataloader:
                    sub_td = sub_td.to(td.device)
                    previous_reward = sub_td["reward"].view(-1, 1)
                    out = self.policy(  # note: remember to clone to avoid in-place replacements!
                        sub_td.clone(),
                        actions=sub_td["action"],
                        env=self.env,
                        return_entropy=True,
                        calc_reward=False
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

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="val", dataloader_idx=dataloader_idx
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="test", dataloader_idx=dataloader_idx
        )

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset, self.train_batch_size, self.shuffle_train_dataloader
        )

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size)

    def on_train_epoch_end(self):
        """Called at the end of the training epoch. This can be used for instance to update the train dataset
        with new data (which is the case in RL).
        """
        # Only update if not in the first epoch
        # If last epoch, we don't need to update since we will not use the dataset anymore
        # if self.current_epoch < self.trainer.max_epochs - 1:
        #     self.train_dataset = self.env.dataset(self.data_cfg["train_data_size"])
        pass

    def _dataloader(self, dataset, batch_size, shuffle=False):
        """Handle both single datasets and list / dict of datasets"""
        if isinstance(dataset, Iterable):
            # load dataloader names if available as dict, else use indices
            if isinstance(dataset, dict):
                self.dataloader_names = list(dataset.keys())
            else:
                self.dataloader_names = [f"{i}" for i in range(len(dataset))]
            # if batch size is int, make it into list
            if isinstance(batch_size, int):
                batch_size = [batch_size] * len(self.dataloader_names)
            assert len(batch_size) == len(
                self.dataloader_names
            ), f"Batch size must match number of datasets. \
                        Found: {len(batch_size)} and {len(self.dataloader_names)}"
            return [
                self._dataloader_single(dset, bsize, shuffle)
                for dset, bsize in zip(dataset.values(), batch_size)
            ]
        else:
            assert isinstance(
                batch_size, int
            ), f"Batch size must be an integer for a single dataset, found {batch_size}"
            return self._dataloader_single(dataset, batch_size, shuffle)

    def _dataloader_single(self, dataset, batch_size, shuffle=False):
        """The dataloader used by the trainer. This is a wrapper around the dataset with a custom collate_fn
        to efficiently handle TensorDicts.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            collate_fn=dataset.collate_fn,
        )