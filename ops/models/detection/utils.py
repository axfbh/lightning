from typing import Any, Union, Sequence
import math
import numpy as np
import torch
from torch import Tensor
from ops.metric.DetectionMetric import MeanAveragePrecision
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler
from utils.nms import non_max_suppression
from lightning import LightningModule, Callback


class Yolo(LightningModule):

    def training_step(self, batch, batch_idx):
        images, targets, shape = batch

        images = images / 255.

        preds = self(images)

        loss, loss_items = self.compute_loss(preds, targets, shape)

        self.log_dict({'box_loss': loss_items[0],
                       'obj_loss': loss_items[1],
                       'cls_loss': loss_items[2]},
                      on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)

        # lightning 的 loss / accumulate ，影响收敛
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def validation_step(self, batch, batch_idx):
        images, targets, shape = batch

        images = images / 255.

        preds, train_out = self.ema_model(images)

        loss = self.compute_loss(train_out, targets, shape)[1]  # box, obj, cls
        if not self.trainer.sanity_checking:
            preds = non_max_suppression(preds,
                                        0.001,
                                        0.6,
                                        labels=[],
                                        max_det=300,
                                        multi_label=False,
                                        agnostic=False)
            self.box_map_metric.update(preds, targets)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            seen, nt, mp, mr, map50, map = self.box_map_metric.compute()

            fitness = map * 0.9 + map50 * 0.1

            self.log_dict({'Images_unplot': seen,
                           'Instances_unplot': nt,
                           'P': mp,
                           'R': mr,
                           'mAP50': map50,
                           'mAP50-95': map,
                           'fitness_un': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)

            self.box_map_metric.reset()

    def configure_model(self) -> None:
        m = self.head  # detection head models
        nl = m.nl  # number of detection layers (to scale hyp)
        nc = m.nc
        self.hyp["box"] *= 3 / nl  # scale to layers
        self.hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
        self.hyp["obj"] *= (max(self.opt.image_size[0],
                                self.opt.image_size[1]) / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.hyp["label_smoothing"] = self.opt.label_smoothing

    def configure_optimizers(self):
        optimizer = smart_optimizer(self,
                                    self.opt.optimizer,
                                    self.hyp['lr'],
                                    self.hyp['momentum'],
                                    self.hyp['weight_decay'])

        scheduler = smart_scheduler(
            optimizer,
            self.opt.scheduler,
            self.current_epoch - 1,
            lrf=self.hyp['lrf'],
            max_epochs=self.trainer.max_epochs
        )

        self.ema_model = ModelEMA(self)

        self.box_map_metric = MeanAveragePrecision(device=self.device)

        return [optimizer], [scheduler]
