"""Shared Lightning scaffolding for GRPO.

`BaseRL` provides data config, dataloaders, metric logging, and the
train/val/test step dispatch. Subclasses implement only the algorithm-specific
parts:

    * configure_optimizers   (which params to train)
    * shared_step            (the loss / update rule)
"""
import torch
from torch import nn
from lightning import LightningModule


class BaseRL(LightningModule):
    def __init__(
        self,
        env,
        policy,
        path_train_data: str,
        path_val_data: str,
        path_test_data: str,
        batch_size: int = 1024,
        metrics: dict = None,
        log_on_step: bool = True,
        train_buckets: list = None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.data_cfg = {
            "batch_size": batch_size,
            "path_train_data": path_train_data,
            "path_val_data": path_val_data,
            "path_test_data": path_test_data,
            "train_buckets": train_buckets,   # None = all buckets
        }
        self.instantiate_metrics(metrics or {})
        self.log_on_step = log_on_step

    # ---- algorithm-specific (subclass must implement) ----------------------
    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    # ---- metrics ------------------------------------------------------------
    def instantiate_metrics(self, metrics: dict):
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])

    def log_metrics(self, metric_dict: dict, phase: str, dataloader_idx=None):
        names = getattr(self, f"{phase}_metrics")
        dataloader_name = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            dataloader_name = "/" + self.dataloader_names[dataloader_idx]
        metrics = {
            f"{phase}/{k}{dataloader_name}": v.mean() if isinstance(v, torch.Tensor) else v
            for k, v in metric_dict.items() if k in names
        }
        log_on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True
        if self.trainer is not None:
            self.log_dict(metrics, on_step=log_on_step, on_epoch=on_epoch, prog_bar=True,
                          sync_dist=True, add_dataloader_idx=False)
        return metrics

    # ---- data ---------------------------------------------------------------
    def load_dataloader(self):
        bs = self.data_cfg["batch_size"]
        buckets = self.data_cfg["train_buckets"]
        self.train_dataset = self.env.dataset(self.data_cfg["path_train_data"],
                                              batch_size=bs, shuffle=True,
                                              buckets=buckets)
        self.val_dataset   = self.env.dataset(self.data_cfg["path_val_data"],
                                              batch_size=bs, shuffle=False,
                                              buckets=buckets)
        self.test_dataset  = self.env.dataset(self.data_cfg["path_test_data"],
                                              batch_size=bs, shuffle=False,
                                              buckets=buckets)

    def setup(self, stage="fit"):
        self.load_dataloader()
        self.dataloader_names = None
        self.setup_loggers()

    def setup_loggers(self):
        if self.loggers:
            hparams_save = {k: v for k, v in self.hparams.items()
                            if not isinstance(v, nn.Module)}
            for logger in self.loggers:
                logger.log_hyperparams(hparams_save)
                logger.log_graph(self)
                logger.save()

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset

    # ---- step dispatch ------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, batch_idx, phase="val", dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, batch_idx, phase="test", dataloader_idx=dataloader_idx)
