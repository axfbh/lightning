from typing import Any

import torch
import torch.nn as nn
from torchvision.models.shufflenetv2 import shufflenet_v2_x2_0
from torchvision.ops.misc import Conv2dNormActivation

from lightning import LightningModule

from ops.models.backbone.utils import _shufflenet_extractor
from ops.models.misc.atrous import ASPP
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler
from ops.metric.SegmentationMetric import Evaluator
from ops.utils.torch_utils import from_torch_to_numpy


class LiteSeg(LightningModule):
    def __init__(self, num_classes):
        super(LiteSeg, self).__init__()
        self.num_classes = num_classes

        self.backbone = _shufflenet_extractor(shufflenet_v2_x2_0(pretrained=True), 6)

        self.aspp = ASPP(2048, 96, [3, 6, 9])

        self.conv1 = Conv2dNormActivation(96 * 5 + 2048, 96, 1)

        self.conv3 = nn.Sequential(Conv2dNormActivation(24 + 96, 96, 3),
                                   Conv2dNormActivation(96, 96, 3))

        self.head = nn.Conv2d(96, num_classes, kernel_size=1)

        self.upsample_4x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_2x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        feat = self.backbone(x)

        x0, x1 = feat['5'], feat['1']
        x0 = torch.cat([x0, self.aspp(x0)], dim=1)
        x0 = self.conv1(x0)
        x0 = self.upsample_4x(x0)
        x = torch.cat([x0, x1], dim=1)
        x = self.conv3(x)
        x = self.head(x)
        x = self.upsample_2x(x)

        return x

    def training_step(self, batch, batch_idx):
        images, masks = batch
        x = self(images)
        loss = self.compute_loss(x, masks) * self.trainer.train_dataloader.batch_size
        self.log('cnt_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.ema_model(images)
        loss = self.compute_loss(preds, masks)

        if not self.trainer.sanity_checking:
            self.mask_metric.add_batch(from_torch_to_numpy(masks), from_torch_to_numpy(preds.argmax(1)))

        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            Acc = self.mask_metric.Pixel_Accuracy()
            Acc_class = self.mask_metric.Pixel_Accuracy_Class()
            mIoU = self.mask_metric.Mean_Intersection_over_Union()
            FWIoU = self.mask_metric.Frequency_Weighted_Intersection_over_Union()

            fitness = mIoU * 0.9 + 0.1 * Acc_class

            self.log_dict({'Acc': Acc,
                           'Acc_class': Acc_class,
                           'mIoU': mIoU,
                           'FWIoU': FWIoU,
                           'fitness_un': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)

            self.mask_metric.reset()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def configure_model(self) -> None:
        self.mask_metric = Evaluator(self.num_classes)

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
