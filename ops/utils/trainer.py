import os
from typing import List, Tuple

import torch
import torch.distributed

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from ops.utils import extract_ip
from ops.utils.callbacks import PlotLogger, TQDMProgressBar


class Trainer:
    def __init__(self,
                 device: str,
                 save_dir: str,
                 names: str,
                 max_epochs: int,
                 bar_train_title: Tuple,
                 bar_val_title: Tuple,
                 gradient_clip_val=10,
                 gradient_clip_algorithm="norm",
                 accumulate: int = 1,
                 num_nodes: int = 1,
                 nproc_per_node: int = 1,
                 master_addr: str = extract_ip(),
                 master_port: str = "8888",
                 node_rank: str = "0",
                 callbacks: List = None):

        ddp = 'auto'

        if nproc_per_node > 1 or num_nodes > 1:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['NODE_RANK'] = node_rank
            ddp = DDPStrategy(process_group_backend="nccl" if torch.distributed.is_nccl_available() else 'gloo')

        tb_logger = TensorBoardLogger(save_dir=save_dir, name=names)

        if callbacks is None:
            callbacks = []

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='fitness_un',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)
        checkpoint_callback.FILE_EXTENSION = '.pt'
        callbacks.append(checkpoint_callback)

        callbacks.append(PlotLogger(len(bar_val_title)))

        callbacks.append(TQDMProgressBar(bar_train_title, bar_val_title))

        self._trainer = L.Trainer(
            accelerator=device,
            devices=nproc_per_node,
            num_nodes=num_nodes,
            logger=tb_logger,
            strategy=ddp,
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=callbacks
        )

    def fit(self, model, train_dataloaders, val_dataloaders, ckpt_path=None):
        self._trainer.fit(model, train_dataloaders, val_dataloaders, ckpt_path=ckpt_path)

    @property
    def trainer(self):
        return self._trainer
