from math import ceil
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

import lightning as pl
from ops.utils.torch_utils import smart_optimizer, smart_scheduler
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score
from ops.utils.torch_utils import ModelEMA
from ops.models.backbone.resnet import ResNet as rn
from lightning.fabric.utilities.rank_zero import rank_zero_only


class ResNet(pl.LightningModule):
    def __init__(self,
                 layers,
                 planes,
                 strides,
                 num_classes,
                 width_per_group=64,
                 group=1):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = rn(layers, planes, strides, num_classes, width_per_group, group)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        preds = self(images)

        loss = self.compute_loss(preds, targets)

        self.log('train_loss', loss,
                 on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        preds = self.ema_model(images)

        loss = self.compute_loss(preds, targets)

        if not self.trainer.sanity_checking:
            self.f1_metric.update(preds, targets)
            self.precision_metric.update(preds, targets)
            self.recall_metric.update(preds, targets)
            self.accuracy_metric.update(preds, targets)

        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            f1_value = self.f1_metric.compute()
            p_value = self.precision_metric.compute()
            r_value = self.recall_metric.compute()
            acc_value = self.accuracy_metric.compute()
            fitness = acc_value
            self.log_dict({'f1': f1_value,
                           'precision': p_value,
                           'recall': r_value,
                           'accuracy': acc_value,
                           'fitness': fitness}, on_epoch=True, sync_dist=True)

    def on_fit_start(self) -> None:
        self.compute_loss = nn.CrossEntropyLoss()
        self.ema_model = ModelEMA(self)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
        super(ResNet, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        ema_step = ceil(self.trainer.num_training_batches * 0.03)
        if batch_idx % ema_step == 0 or batch_idx == self.trainer.num_training_batches:
            self.ema_model.update(self)

    def configure_model(self) -> None:
        """
        This is section, model not to(device)
        :return:
        """
        self.accuracy_metric = Accuracy('multiclass', average='weighted', num_classes=self.num_classes)
        self.precision_metric = Precision('multiclass', average='weighted', num_classes=self.num_classes)
        self.recall_metric = Recall('multiclass', average='weighted', num_classes=self.num_classes)
        self.f1_metric = F1Score('multiclass', average='weighted', num_classes=self.num_classes)

    def configure_optimizers(self):
        """
        First move, model.to(device)
        :return:
        """
        optimizer = smart_optimizer(self,
                                    self.optim,
                                    self.hyp['lr'],
                                    self.hyp['momentum'],
                                    self.hyp['weight_decay'])

        scheduler = smart_scheduler(optimizer,
                                    self.sche,
                                    self.current_epoch - 1,
                                    T_max=self.trainer.max_epochs)

        return [optimizer], [scheduler]
