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
from lightning.pytorch.callbacks import ProgressBar


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
                 nbs: int,
                 momentum: float,
                 warmup_epoch: int,
                 warmup_bias_lr: float,
                 warmup_momentum: float):
        self.nbs = nbs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_epoch = warmup_epoch
        self.momentum = momentum

    def _update_warmup_parameters(self, trainer, epoch, batch_idx):
        ni = batch_idx + trainer.num_training_batches * epoch
        nw = max(round(trainer.num_training_batches * self.warmup_epoch), 100)
        batch_size = trainer.train_dataloader.batch_size
        return ni, nw, batch_size

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        optimizer = pl_module.optimizers()
        scheduler = pl_module.lr_schedulers()
        epoch = pl_module.current_epoch

        ni, nw, batch_size = self._update_warmup_parameters(trainer, epoch, batch_idx)

        if ni <= nw:
            xi = [0, nw]  # x interp
            trainer.accumulate_grad_batches = max(1, np.interp(ni, xi, [1, self.nbs / batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                lf = scheduler.lr_lambdas[j]
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * lf(epoch)]
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        ni,
                        xi,
                        [self.warmup_momentum, self.momentum]
                    )


class TQDMProgressBar(tqdm_progress.TQDMProgressBar):
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format

    def __init__(self, train_title, val_title, refresh_rate: int = 1, process_position: int = 0):
        super(TQDMProgressBar, self).__init__(refresh_rate, process_position)
        self.train_title = train_title
        self.val_title = val_title

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=self.TQDM_BAR_FORMAT,
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
            bar_format=self.TQDM_BAR_FORMAT,
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
            bar_format=self.TQDM_BAR_FORMAT,
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
        return _info(
            ("\n" + "%11s" * 3) % ("Epoch", "GPU_mem", "lr") + ("%11s" * len(self.train_title)) % self.train_title
        )

    def get_val_tile(self) -> str:
        return ("%11s" * len(self.val_title)) % self.val_title

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

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        n = trainer._active_loop.batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):  # rank 0 更新 bar
            _update_n(self.train_progress_bar, n)
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            lr = pl_module.optimizers().optimizer.param_groups[0]['lr']
            desc = ("%11i" + "%11s" + "%11.4g") % (trainer.current_epoch, mem, lr) + self.get_description(trainer,
                                                                                                          pl_module)
            self.train_progress_bar.set_description(desc)

        if self.trainer.is_last_batch:
            self.train_progress_bar.close()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int
                           ) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

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


class LitProgressBar(ProgressBar):

    def __init__(self, refresh_rate=1):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.refresh_rate = refresh_rate
        self.prev_mean = None

    def disable(self):
        self.enable = False

    def get_meters(self, trainer, pl_module):
        meters = self.get_metrics(trainer, pl_module)

        meters_mean = self.get_meters_mean(trainer)

        meters.pop('v_num')

        loss_str = []

        for name, meter in meters.items():
            if name.endswith('step'):
                name = name[:-5]
                if name.startswith('loss'):
                    if self.prev_mean is None:
                        loss_str.append(
                            "{}: {:.4f} ({:.4f})".format(name, meter, meters_mean[name])
                        )
                    else:
                        loss_str.append(
                            "{}: {:.4f} ({:.4f}-{:.4f})".format(name, meter, self.prev_mean[name], meters_mean[name])
                        )
                else:
                    loss_str.append(
                        "{}: {:.4f} ".format(name, meter)
                    )

        return '  '.join(loss_str)

    def get_meters_mean(self, trainer) -> dict:
        meters_name = {}
        for result in trainer._results.result_metrics:
            name = result.meta.name
            value = result.value
            cumulated_batch_size = result.cumulated_batch_size
            meters_name[name] = (value / cumulated_batch_size).item()
        return meters_name

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)

        if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_train_batches - 1):
            MB = 1024.0 * 1024.0 * 1024.0

            delimiter = '  '

            space_fmt = ':' + str(len(str(batch_idx))) + 'd'

            log_msg = delimiter.join([
                'Epoch: [{}]'.format(trainer.current_epoch),
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'max mem: {memory:.2f} GB'
            ])

            print(log_msg.format(batch_idx,
                                 self.total_train_batches,
                                 meters=self.get_meters(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prev_mean = self.get_meters_mean(trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super(LitProgressBar, self).on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                                            dataloader_idx)
        if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_val_batches - 1):
            MB = 1024.0 * 1024.0 * 1024.0

            delimiter = '  '

            space_fmt = ':' + str(len(str(batch_idx))) + 'd'

            log_msg = delimiter.join([
                'Val: [{}]'.format(trainer.current_epoch),
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'max mem: {memory:.2f} GB'
            ])

            print(log_msg.format(batch_idx,
                                 self.total_val_batches,
                                 meters=self.get_meters(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prev_mean = self.get_meters_mean(trainer)
