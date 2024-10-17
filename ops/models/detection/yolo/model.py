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
    def __init__(self, cfg, weight=None):
        self.weight = weight
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

    def train(self,
              *,
              data,
              epochs=100,
              batch=16,
              imgsz=640,
              device: Union[List[int], str, int] = 1,
              workers=8,
              project='runs',
              name='train',
              optimizer='SGD',
              resume=False,
              lr=0.01,
              lrf=0.01,
              momentum=0.937,
              weight_decay=0.0005,
              warmup_epochs=3,
              warmup_momentum=0.8,
              warmup_bias_lr=0.1,
              box=7.5,
              cls=0.5,
              dfl=1.5,
              nbs=64,
              degrees: float = 0.0,
              translate: float = 0.1,
              scale: float = 0.5,
              shear: float = 0.0,
              flipud: float = 0.0,
              fliplr: float = 0.5,
              mosaic: float = 1.0,
              master_addr: str = extract_ip(),
              master_port: str = "8888",
              node_rank: str = "0",
              num_nodes=1
              ):
        # ------------ hyp-parameter ------------
        rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

        # ------------- data -------------
        data = OmegaConf.load(data)

        train_dataloader = create_dataloader(data.train,
                                             imgsz,
                                             batch,
                                             data.names,
                                             degrees,
                                             translate,
                                             scale,
                                             flipud,
                                             fliplr,
                                             mosaic,
                                             image_set='car_train',
                                             augment=True,
                                             workers=workers,
                                             shuffle=True,
                                             persistent_workers=True)

        val_dataloader = create_dataloader(data.val,
                                           imgsz,
                                           batch * 2,
                                           data.names,
                                           image_set='car_val',
                                           augment=False,
                                           workers=workers,
                                           shuffle=False,
                                           persistent_workers=True)
        params = self.params
        params.update({
            'imgsz': imgsz,
            'optim': optimizer
        })
        model = self.model(**params)

        accelerator = device if device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        bar_train_title = ("box_loss", "obj_loss", "cls_loss")
        bar_val_title = ("Images", "Instances", "P", "R", "mAP50", "mAP50-95")

        # ---------- batch size 参数 ----------
        # batch_size = batch
        # nbs = 64  # nominal batch size
        # accumulate = max(round(nbs / batch_size), 1)
        # hyp["weight_decay"] *= batch_size * accumulate / nbs

        warmup_callback = WarmupLR(nbs=nbs,
                                   momentum=momentum,
                                   warmup_bias_lr=warmup_bias_lr,
                                   warmup_epoch=warmup_epochs,
                                   warmup_momentum=warmup_momentum)

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
            devices=device,
            num_nodes=num_nodes,
            logger=TensorBoardLogger(save_dir=f'./{project}', name=name),
            strategy=auto_distribute(num_nodes, device, master_addr, master_port, node_rank),
            max_epochs=epochs,
            # accumulate_grad_batches=accumulate,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=[warmup_callback, checkpoint_callback, plot_callback, progress_bar_callback]
        )

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.weight if resume else None)

    @property
    def trainer(self):
        return self._trainer
