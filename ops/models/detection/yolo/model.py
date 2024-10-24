from typing import Any
import torch

from lightning import LightningModule

from ops.metric.DetectionMetric import DetectionMetric
from ops.utils.torch_utils import ModelEMA, smart_optimizer
from ops.utils.torch_utils import one_linear

from utils.nms import non_max_suppression


class YoloModel(LightningModule):

    def __init__(self, hyp):
        super(YoloModel, self).__init__()
        self.hyp = hyp

    def forward(self, x):
        return self(x)

    def training_step(self, batch, batch_idx):
        batch["img"] = batch["img"] / 255.

        preds = self(batch["img"])

        loss, loss_items = self.criterion(preds, batch)

        self.log_dict({'box_loss': loss_items[0],
                       'obj_loss': loss_items[1],
                       'cls_loss': loss_items[2]},
                      on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)

        # lightning 的 loss / accumulate ，影响收敛
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def validation_step(self, batch, batch_idx):
        batch["img"] = batch["img"] / 255.

        preds, train_out = self.ema_model(batch["img"])

        loss = self.criterion(train_out, batch)[1]  # box, obj, cls
        if not self.trainer.sanity_checking:
            preds = non_max_suppression(preds,
                                        0.001,
                                        0.6,
                                        labels=[],
                                        max_det=300,
                                        multi_label=False,
                                        agnostic=False)
            self.metric.update(preds, batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            seen, nt, mp, mr, map50, map = self.metric.compute()

            fitness = map * 0.9 + map50 * 0.1

            self.log_dict({'Images_unplot': seen,
                           'Instances_unplot': nt,
                           'P': mp,
                           'R': mr,
                           'mAP50': map50,
                           'mAP50-95': map,
                           'fitness_un': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)

            self.metric.reset()

    def configure_model(self) -> None:
        m = self.head  # detection head models
        nl = m.nl  # number of detection layers (to scale hyp)
        nc = m.nc

        self.hyp['box'] *= 3 / nl  # scale to layers
        self.hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        self.hyp['obj'] *= (max(
            self.hyp['imgsz'][0],
            self.hyp['imgsz'][1]
        ) / 640) ** 2 * 3 / nl  # scale to image size and layers

        batch_size = self.hyp.batch
        nbs = self.hyp.nbs  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)
        self.hyp['weight_decay'] *= batch_size * accumulate / nbs

        self.metric = DetectionMetric(device=self.device, background=False)

    def configure_optimizers(self):
        optimizer = smart_optimizer(self,
                                    self.hyp['optimizer'],
                                    self.hyp['lr0'],
                                    self.hyp['momentum'],
                                    self.hyp['weight_decay'])

        fn = one_linear(lrf=self.hyp['lrf'], max_epochs=self.hyp['epochs'])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      last_epoch=self.current_epoch - 1,
                                                      lr_lambda=fn)

        self.ema_model = ModelEMA(self)

        return [optimizer], [scheduler]
