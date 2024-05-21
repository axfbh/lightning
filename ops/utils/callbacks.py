import os
import sys

from typing import Any, Dict, Union

import numpy as np
import matplotlib.pyplot as plt

import torch

import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import _info
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.progress.tqdm_progress import _update_n, convert_inf, Tqdm
from lightning.pytorch.callbacks.progress import tqdm_progress


class PlotLogger(Callback):

    def __init__(self, num):
        super(PlotLogger, self).__init__()
        self.values = [[] for _ in range(num)]

    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            for i, result in enumerate(trainer._results.result_metrics):
                name = result.meta.name
                value = result.value
                cumulated_batch_size = result.cumulated_batch_size
                if not name.endswith('_un') and not name.endswith('_unplot'):
                    self.values[i].append((value / cumulated_batch_size).item())
                    plt.plot(np.array(self.values[i]), 'b', linewidth=2, label=name)
                    plt.grid(True)
                    plt.xlabel('Epoch')
                    plt.ylabel(name)
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join(trainer.logger.log_dir, f"{name}.png"))
                    plt.cla()
                    plt.close("all")


class WarmupLR(Callback):
    def __init__(self,
                 momentum: float,
                 warmup_epoch: int,
                 warmup_bias_lr: float,
                 warmup_momentum: float):
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_epoch = warmup_epoch
        self.momentum = momentum

    def on_train_batch_start(self,
                             trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule",
                             batch: Any,
                             batch_idx: int) -> None:
        ni = trainer.global_step
        warmup_iter = self.warmup_epoch * (trainer.num_training_batches + 1)
        if ni <= warmup_iter:
            xi = [0, warmup_iter]  # x interp
            for i, optimizer in enumerate(trainer.optimizers):
                for j, x in enumerate(optimizer.param_groups):
                    new_lr = trainer.lr_scheduler_configs[i].scheduler._get_closed_form_lr()[j]
                    # new_lr = 0.01
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [self.warmup_bias_lr if j == 0 else 0.0, new_lr]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni,
                            xi,
                            [self.warmup_momentum, self.momentum]
                        )


TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format


class TQDMProgressBar(tqdm_progress.TQDMProgressBar):

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=TQDM_BAR_FORMAT,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=TQDM_BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=TQDM_BAR_FORMAT,
        )

    def get_description(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> str:
        res = []
        for result in trainer._results.result_metrics:
            name = result.meta.name
            value = result.value
            cumulated_batch_size = result.cumulated_batch_size
            if not name.endswith('_un') and not name.endswith('_unprint'):
                res.append((value / cumulated_batch_size).item())
        return ("%11.4g" * len(res)) % (*res,)

    def get_train_tile(self) -> None:
        raise NotImplemented

    def get_val_tile(self) -> str:
        raise NotImplemented

    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()

    def on_sanity_check_end(self, *_: Any) -> None:
        self.val_progress_bar.close()

    def on_train_start(self, *_: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self.is_enabled:  # rank 0 打印
            self.get_train_tile()

        self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int
                           ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):  # rank 0 更新 bar
            _update_n(self.train_progress_bar, n)
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            lr = trainer.optimizers[0].param_groups[0]['lr']
            desc = ("%11i" + "%11s" + "%11.4g") % (trainer.current_epoch, mem, lr) + self.get_description(trainer,
                                                                                                          pl_module)
            self.train_progress_bar.set_description(desc)

        if n == self.train_progress_bar.total:
            self.train_progress_bar.close()

    def on_train_end(self, *_: Any) -> None:
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.desc = self.get_val_tile()

    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                  batch: Any,
                                  batch_idx: int,
                                  dataloader_idx: int = 0,
                                  ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
        self.val_progress_bar.initial = 0

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0,
                                ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
        if not trainer.sanity_checking and self.is_enabled:  # rank 0 打印
            _info(self.get_description(trainer, pl_module))
