import os
from typing import Union, List
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.utilities.rank_zero import rank_zero_info

from ops.utils import extract_ip
from ops.utils.callbacks import WarmupLR
from ops.utils.callbacks import PlotLogger, TQDMProgressBar
from ops.utils.logging import colorstr
from ops.utils.torch_utils import auto_distribute
from ops.models.detection.yolo import YoloV5, YoloV4, YoloV7, YoloV8

from dataloader import create_dataloader


class Yolo:
    def __init__(self, model, weight=None):
        model = OmegaConf.load(model)
        version = model.version
        phi = model.phi
        num_classes = model.nc
        anchors = model.anchors

        self.weight = weight

        self.model = {
            'yolov3': YoloV4,
            'yolov4': YoloV4,
            'yolov5': YoloV5,
            'yolov7': YoloV7,
            'yolov8': YoloV8,
        }[version]

        self.params = {
            'yolov3': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov4': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov5': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov7': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov8': {'num_classes': num_classes, 'phi': phi},
        }[version]

    def train(self,
              *,
              data,
              master_addr: str = extract_ip(),
              master_port: str = "8888",
              node_rank: str = "0",
              num_nodes=1,
              **kwargs):
        # ------------ hyp-parameter ------------
        hyp = OmegaConf.load('./cfg/default.yaml')
        hyp.update(kwargs)
        rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

        # ------------ model ------------
        params = self.params
        params.update({
            'hyp': hyp,
        })
        model = self.model(**params)

        # ------------ data ------------
        data = OmegaConf.load(data)

        train_dataloader = create_dataloader(data.train,
                                             hyp.batch,
                                             data.names,
                                             hyp,
                                             image_set='car_train',
                                             augment=True,
                                             workers=hyp.workers,
                                             shuffle=True,
                                             persistent_workers=True)

        val_dataloader = create_dataloader(data.val,
                                           hyp.batch * 2,
                                           data.names,
                                           hyp,
                                           image_set='car_val',
                                           augment=False,
                                           workers=hyp.workers,
                                           shuffle=False,
                                           persistent_workers=True)

        accelerator = hyp.device if hyp.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        bar_train_title = ("box_loss", "obj_loss", "cls_loss")
        bar_val_title = ("Images", "Instances", "P", "R", "mAP50", "mAP50-95")

        warmup_callback = WarmupLR(nbs=hyp.nbs,
                                   momentum=hyp.momentum,
                                   warmup_bias_lr=hyp.warmup_bias_lr,
                                   warmup_epoch=hyp.warmup_epochs,
                                   warmup_momentum=hyp.warmup_momentum)

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='fitness_un',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)
        checkpoint_callback.FILE_EXTENSION = '.pt'

        plot_callback = PlotLogger(len(bar_val_title))

        progress_bar_callback = TQDMProgressBar(bar_train_title, bar_val_title)

        self._trainer = L.Trainer(
            accelerator=accelerator,
            devices=hyp.device,
            num_nodes=num_nodes,
            logger=TensorBoardLogger(save_dir=f'./{hyp.project}', name=hyp.name),
            strategy=auto_distribute(num_nodes, hyp.device, master_addr, master_port, node_rank),
            max_epochs=hyp.epochs,
            # accumulate_grad_batches=accumulate,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=[warmup_callback, checkpoint_callback, plot_callback, progress_bar_callback]
        )

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.weight if hyp.resume else None)

    @property
    def trainer(self):
        return self._trainer
