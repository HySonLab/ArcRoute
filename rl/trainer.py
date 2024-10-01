import torch
from lightning import Trainer

class RL4COTrainer(Trainer):
    def __init__(
        self,
        accelerator = "auto",
        callbacks = None,
        logger = None,
        min_epochs = None,
        max_epochs = None,
        strategy = "auto",
        devices = "auto",
        precision = "16-mixed",
        reload_dataloaders_every_n_epochs: int = 1,
        disable_profiling_executor: bool = True,
        matmul_precision = "medium",
        **kwargs,
    ):
        # Reference: https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
        if disable_profiling_executor:
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        # Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)

        # Main call to `Trainer` superclass
        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            strategy=strategy,
            devices=devices,
            precision=precision,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            **kwargs,
        )