from typing import Any
import torch

from lightning import LightningModule

from ops.metric.DDectectionMetric import CocoEvaluator
from ops.utils.torch_utils import ModelEMA, smart_optimizer
from ops.utils.torch_utils import one_linear
from ops.models.detection.detr.postprocess import PostProcess
import torchvision


def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


class DetrModel(LightningModule):

    def __init__(self, hyp):
        super(DetrModel, self).__init__()
        self.hyp = hyp

    def forward(self, x):
        return self(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        imgs.tensors = imgs.tensors / 255.
        preds = self(imgs)

        loss_dict = self.criterion(preds, targets)  # box, obj, cls
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self.log_dict({'loss_ce': loss_dict['loss_ce'].item(),
                       'loss_bbox': loss_dict['loss_bbox'].item(),
                       'loss_giou': loss_dict['loss_giou'].item()},
                      on_epoch=True, sync_dist=True, batch_size=self.trainer.train_dataloader.batch_size)

        # lightning 的 loss / accumulate ，影响收敛
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        imgs.tensors = imgs.tensors / 255.
        train_out = self.ema_model(imgs)

        loss_dict = self.criterion(train_out, targets)  # box, obj, cls
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not self.trainer.sanity_checking:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            pred = self.postprocesser(train_out, orig_target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, pred)}
            self.metric.update(res)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.metric.accumulate()
            self.metric.summarize()
            stats = self.metric.coco_eval['bbox'].stats.tolist()
            # seen, nt, mp, mr, map50, map = self.metric.compute()
            #
            # fitness = map * 0.9 + map50 * 0.1
            #
            # self.log_dict({'Images_unplot': seen,
            #                'Instances_unplot': nt,
            #                'P': mp,
            #                'R': mr,
            #                'mAP50': map50,
            #                'mAP50-95': map,
            #                'fitness_un': fitness},
            #               on_epoch=True, sync_dist=True, batch_size=self.trainer.val_dataloaders.batch_size)
            #
            # self.metric.reset()

    def configure_model(self) -> None:
        batch_size = self.hyp.batch
        nbs = self.hyp.nbs  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)
        self.hyp['weight_decay'] *= batch_size * accumulate / nbs

        base_ds = get_coco_api_from_dataset(self.val_dataset)

        self.metric = CocoEvaluator(base_ds, ['bbox'])

        self.postprocesser = PostProcess()

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
