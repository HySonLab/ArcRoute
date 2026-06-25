"""Shared Lightning scaffolding for GRPO.

`BaseRL` provides data config, dataloaders, metric logging, the train/val/test
step dispatch, and the per-epoch data reload. Subclasses implement only the
algorithm-specific parts:

    * configure_optimizers   (which params to train)
    * shared_step            (the loss / update rule)
"""
import os

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
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        test_data_size: int = 1000,
        metrics: dict = None,
        log_on_step: bool = True,
        shuffle_train_dataloader: bool = True,
        dataloader_num_workers: int = 24,
        reload_train_dataloader: int = 4,
    ):
        super().__init__()
        self.env = env
        self.policy = policy

        if batch_size > train_data_size or batch_size > val_data_size:
            print("batch_size should be less than or equal to train_data_size.")
            batch_size = min(train_data_size, val_data_size)

        self.data_cfg = {
            "batch_size": batch_size,
            "val_batch_size": batch_size,
            "test_batch_size": batch_size,
            "train_data_size": train_data_size,
            "val_data_size": val_data_size,
            "test_data_size": test_data_size,
            "path_train_data": path_train_data,
            "path_val_data": path_val_data,
            "path_test_data": path_test_data,
        }
        self.instantiate_metrics(metrics or {})
        self.log_on_step = log_on_step
        self.shuffle_train_dataloader = shuffle_train_dataloader
        self.dataloader_num_workers = dataloader_num_workers
        self.reload_train_dataloader = reload_train_dataloader

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
        self.log_on_step = metrics.get("log_on_step", True)

    def log_metrics(self, metric_dict: dict, phase: str, dataloader_idx=None):
        """Log metrics to logger and progress bar."""
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
        self.log_dict(metrics, on_step=log_on_step, on_epoch=on_epoch, prog_bar=True,
                      sync_dist=True, add_dataloader_idx=False)
        return metrics

    # ---- data ---------------------------------------------------------------
    def load_dataloader(self):
        self.train_dataset = self.env.dataset(self.data_cfg["train_data_size"],
                                              batch_size=self.data_cfg["batch_size"], shuffle=True,
                                              data=self.data_cfg["path_train_data"],
                                              num_workers=self.dataloader_num_workers)
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"],
                                            batch_size=self.data_cfg["batch_size"], shuffle=False,
                                            data=self.data_cfg["path_val_data"],
                                            num_workers=self.dataloader_num_workers)
        self.test_dataset = self.env.dataset(self.data_cfg["test_data_size"],
                                             batch_size=self.data_cfg["batch_size"], shuffle=False,
                                             data=self.data_cfg["path_test_data"],
                                             num_workers=self.dataloader_num_workers)

    def setup(self, stage="fit"):
        print(">>>>>>>>>>>>>")
        print(f"train_data_size={self.data_cfg['train_data_size']}   bs={self.data_cfg['batch_size']}")
        print(f"val_data_size={self.data_cfg['val_data_size']}   bs={self.data_cfg['batch_size']}")
        print(f"test_data_size={self.data_cfg['test_data_size']}   bs={self.data_cfg['batch_size']}")
        self.load_dataloader()
        self.dataloader_names = None
        self.setup_loggers()

    def setup_loggers(self):
        """Log all hyperparameters except those in `nn.Module`."""
        if self.loggers is not None:
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

    def on_train_epoch_end(self):
        # Step a MultiStepLR scheduler if one is configured.
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.MultiStepLR):
            sch.step()
        # Regenerate fresh training data every `reload_train_dataloader` epochs.
        if (self.current_epoch + 1) % self.reload_train_dataloader == 0:
            path = self.data_cfg["path_train_data"]
            # Only delete generated cache files (under data/), not external paths.
            if path.startswith("data/") and os.path.exists(path):
                print("DELETED train_data.pt")
                os.remove(path)
            self.load_dataloader()
