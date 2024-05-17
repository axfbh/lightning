from math import ceil
from typing import List, Any

import torch.nn as nn
import torch

from ops.models.neck.spp import SPPCSPC
from ops.models.head.yolo_head import YoloV7Head
from ops.models.backbone.elandarknet import ElanDarkNet, CBS, MP1, Elan
from ops.models.backbone.utils import _elandarknet_extractor
from ops.loss.yolo_loss import YoloLossV7
from ops.utils.torch_utils import ModelEMA, smart_optimizer, smart_scheduler

import lightning as pl
from lightning import LightningModule
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ops.metric.DetectionMetric import MeanAveragePrecision


class YoloV7(LightningModule):
    def __init__(self, anchors: List, num_classes: int, phi: str):
        super(YoloV7, self).__init__()

        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]

        self.backbone = _elandarknet_extractor(ElanDarkNet(transition_channels, block_channels, n, phi), 5)

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16,
                               conv_layer=CBS,
                               activation_layer=nn.SiLU)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_P5 = CBS(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = CBS(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Elan(transition_channels * 16, panet_channels * 4,
                                        transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4 = CBS(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = CBS(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Elan(transition_channels * 8, panet_channels * 2,
                                        transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1 = MP1(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Elan(transition_channels * 16, panet_channels * 4,
                                          transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2 = MP1(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Elan(transition_channels * 32, panet_channels * 8,
                                          transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_1 = CBS(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = CBS(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = CBS(transition_channels * 16, transition_channels * 32, 3, 1)

        self.head = YoloV7Head([transition_channels * 8, transition_channels * 16, transition_channels * 32],
                               anchors,
                               num_classes)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backbone(x)

        feat1, feat2, feat3 = x['0'], x['1'], x['2']

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        return self.head([P3, P4, P5], H, W)

    def training_step(self, batch, batch_idx):
        images, targets, shape = batch

        images = images / 255.

        image_size = torch.as_tensor(shape, device=self.device)

        preds = self(images)

        loss, loss_items = self.compute_loss(preds, targets, image_size)

        self.log_dict({'box_loss': loss_items[0],
                       'obj_loss': loss_items[1],
                       'cls_loss': loss_items[2]},
                      on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, shape = batch

        images = images / 255.

        image_size = torch.as_tensor(shape, device=self.device)

        preds, train_out = self(images)

        loss = self.compute_loss(train_out, targets, image_size)[1]  # box, obj, cls

        if not self.trainer.sanity_checking:
            self.map_metric.update(preds, targets)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            seen, nt, mp, mr, map50, map = self.map_metric.compute()

            fitness = map * 0.9 + map50 * 0.1

            self.log_dict({'Images': seen,
                           'Instances': nt,
                           'P': mp,
                           'R': mr,
                           'mAP50': map50,
                           'mAP50-95': map,
                           'fitness': fitness},
                          on_epoch=True, sync_dist=True, batch_size=self.trainer.num_val_batches[0])

            self.map_metric.reset()

    def on_fit_start(self) -> None:
        self.compute_loss = YoloLossV7(self)
        self.ema_model = ModelEMA(self)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
        super(YoloV7, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        ema_step = ceil(self.trainer.num_training_batches * 0.03)
        if batch_idx % ema_step == 0 or batch_idx == self.trainer.num_training_batches:
            self.ema_model.update(self)

    def configure_model(self) -> None:
        self.map_metric = MeanAveragePrecision(device=self.device, conf_thres=0.001, iou_thres=0.6, max_det=300)

    def configure_optimizers(self):
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
