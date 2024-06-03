from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial
from lightning import LightningModule
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler
from ops.loss.dice_loss import DiceLoss

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

        self.out = nn.Conv2d(base_channels, num_classes, 1, 1, 0)

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

        return self.out(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        x = self(images)
        x = x.permute([0, 2, 3, 1]).contiguous()
        loss = self.compute_loss(x, masks)
        self.log('dice_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)
        return loss

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
        self.compute_loss = DiceLoss(20, True)
