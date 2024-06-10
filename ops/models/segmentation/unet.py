from typing import Any

import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial
from lightning import LightningModule
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler
from ops.metric.SegmentationMetric import Evaluator
from ops.utils.torch_utils import from_torch_to_numpy

CBR = partial(Conv2dNormActivation, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)


def make_two_conv(filters_list, in_filters):
    m = nn.Sequential(
        CBR(in_filters, filters_list[0], 3),
        CBR(filters_list[0], filters_list[1], 3),
    )
    return m


class Unet(LightningModule):
    def __init__(self, in_channels=3, base_channels=64, base_depth=4, num_classes=20):
        super(Unet, self).__init__()
        self.num_classes = num_classes
        self.down_module = nn.ModuleList()
        self.up_module = nn.ModuleList()
        self.up_conv_module = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2, 2)

        for i in range(base_depth):
            self.down_module.append(make_two_conv([base_channels, base_channels], in_channels))
            in_channels = base_channels
            base_channels *= 2

        self.down_module.append(make_two_conv([base_channels, base_channels], in_channels))

        for i in range(base_depth):
            in_channels = base_channels
            base_channels //= 2
            self.up_conv_module.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                CBR(in_channels, base_channels, 1)
            ))
            self.up_module.append(make_two_conv([base_channels, base_channels], in_channels))

        self.head = nn.Conv2d(base_channels, self.num_classes, 1, 1, 0)

    def forward(self, x):
        sampling = []
        for conv in self.down_module:
            sampling.append(conv(x))
            x = sampling[-1]
            x = self.maxpool(x)

        x = sampling.pop(-1)

        for conv, up in zip(self.up_module, self.up_conv_module):
            x4 = sampling.pop(-1)
            x = up(x)
            x = conv(torch.cat([x4, x], 1))

        return self.head(x)

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
        self.mask_metric = Evaluator(self.n_classes)

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
