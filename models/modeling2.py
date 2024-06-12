from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from ops.models.backbone.resnet import ResNet
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler


class RModle(LightningModule):
    def __init__(self,
                 layers,
                 planes,
                 strides,
                 num_classes,
                 width_per_group=64,
                 group=1):
        super(RModle, self).__init__()

        self.backbone = ResNet(layers, planes, strides, num_classes, width_per_group, group)
        self.arr = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        x = self(images)
        loss = self.compute_loss(x, targets) * self.trainer.train_dataloader.batch_size
        self.log('cnt_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.ema_model(images)
        loss = self.compute_loss(preds, targets)
        self.arr.append(torch.mean((torch.argmax(F.softmax(preds, dim=-1), dim=-1) == targets).type(torch.FloatTensor)))
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            acc = torch.mean(torch.tensor(self.arr, device=self.device))
            fitness = acc.item()

            self.log_dict({'Acc': acc,
                           'fitness_un': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)

            self.arr.clear()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

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

        return [optimizer], [scheduler]

    def on_fit_start(self) -> None:
        self.compute_loss = nn.CrossEntropyLoss(ignore_index=255)
