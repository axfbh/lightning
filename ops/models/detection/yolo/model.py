import os
from typing import Union, List
from omegaconf import OmegaConf

import torch
import torch.distributed

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.fabric.utilities.rank_zero import rank_zero_info

from ops.utils import extract_ip
from ops.utils.callbacks import WarmupLR
from ops.utils.callbacks import PlotLogger, TQDMProgressBar
from ops.utils.logging import colorstr

from ops.models.detection.yolo import YoloV5, YoloV4, YoloV7, YoloV8

from dataloader import create_dataloader


class Yolo:
    def __init__(self, cfg):
        cfg = OmegaConf.load(cfg)

        model = cfg.model
        phi = cfg.phi
        num_classes = cfg.nc
        anchors = cfg.anchors

        self.model = {
            'yolov3': YoloV4,
            'yolov4': YoloV4,
            'yolov5': YoloV5,
            'yolov7': YoloV7,
            'yolov8': YoloV8,
        }[model]

        self.params = {
            'yolov3': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov4': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov5': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov7': {'anchors': anchors, 'num_classes': num_classes, 'phi': phi},
            'yolov8': {'num_classes': num_classes, 'phi': phi},
        }[model]

    def fit(self,
            data,
            *,
            epochs=None,
            hyp=None,
            imgsz=None,
            batch=None,
            workers=None,
            optimizer='SGD',
            weight=None,
            device: Union[List[int], str, int] = 1,
            project='runs',
            name='train',
            master_addr: str = extract_ip(),
            master_port: str = "8888",
            node_rank: str = "0",
            num_nodes=1,
            resume=None):
        hyp = OmegaConf.load('./data/hyp/hyp-yolo-low.yaml' if hyp is None else hyp)
        rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

        data = OmegaConf.load(data)

        train_dataloader = create_dataloader(data.train,
                                             imgsz,
                                             batch,
                                             data.names,
                                             hyp=hyp,
                                             image_set='car_train',
                                             augment=True,
                                             workers=workers,
                                             shuffle=True,
                                             persistent_workers=True)

        val_dataloader = create_dataloader(data.val,
                                           imgsz,
                                           batch * 2,
                                           data.names,
                                           hyp=hyp,
                                           image_set='car_val',
                                           augment=False,
                                           workers=workers,
                                           shuffle=False,
                                           persistent_workers=True)
        params = self.params
        params.update({'hyp': hyp,
                       'imgsz': imgsz,
                       'batch': batch,
                       'optim': optimizer})
        model = self.model(**params)

        ddp = 'auto'
        bar_train_title = ("box_loss", "obj_loss", "cls_loss"),
        bar_val_title = ("Images", "Instances", "P", "R", "mAP50", "mAP50-95"),
        accelerator = device if device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        # ---------- batch size 参数 ----------
        batch_size = batch
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)
        hyp["weight_decay"] *= batch_size * accumulate / nbs

        if num_nodes > 1 or (isinstance(device, List) and len(device) > 1):
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['NODE_RANK'] = node_rank
            ddp = DDPStrategy(process_group_backend="nccl" if torch.distributed.is_nccl_available() else 'gloo')

        tb_logger = TensorBoardLogger(save_dir=f'./{project}', name=name)

        callbacks = []

        warmup_callback = WarmupLR(nbs=nbs,
                                   momentum=hyp['momentum'],
                                   warmup_bias_lr=hyp['warmup_bias_lr'],
                                   warmup_epoch=hyp["warmup_epoch"],
                                   warmup_momentum=hyp['warmup_momentum'])
        callbacks.append(warmup_callback)

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
            accelerator=accelerator,
            devices=device,
            num_nodes=num_nodes,
            logger=tb_logger,
            strategy=ddp,
            max_epochs=epochs,
            accumulate_grad_batches=accumulate,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=callbacks
        )

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=weight if resume else None)

    @property
    def trainer(self):
        return self._trainer
