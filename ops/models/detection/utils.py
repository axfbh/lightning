from typing import Any
import torch
import math
from ops.metric.DetectionMetric import MeanAveragePrecision
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler

from lightning import LightningModule


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

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, shape = batch

        images = images / 255.

        preds, train_out = self.ema_model(images)

        loss = self.compute_loss(train_out, targets, shape)[1]  # box, obj, cls

        if not self.trainer.sanity_checking:
            self.map_metric.update(preds, targets)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            seen, nt, mp, mr, map50, map = self.map_metric.compute()

            fitness = map * 0.9 + map50 * 0.1

            self.log_dict({'Images_unplot': seen,
                           'Instances_unplot': nt,
                           'P': mp,
                           'R': mr,
                           'mAP50': map50,
                           'mAP50-95': map,
                           'fitness_un': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)

            self.map_metric.reset()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def configure_model(self) -> None:
        self.map_metric = MeanAveragePrecision(device=self.device, conf_thres=0.001, iou_thres=0.6, max_det=300)

    def configure_optimizers(self):
        optimizer = smart_optimizer(self,
                                    self.optim,
                                    self.hyp['lr'],
                                    self.hyp['momentum'],
                                    self.hyp['weight_decay'])

        scheduler = smart_scheduler(
            optimizer,
            self.sche,
            self.current_epoch - 1,
            T_max=self.trainer.max_epochs
        )

        self.ema_model = ModelEMA(self)

        return [optimizer], [scheduler]
