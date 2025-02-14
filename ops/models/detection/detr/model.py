from typing import Any
import torch

from lightning import LightningModule

from ops.metric.DDectectionMetric import CocoEvaluator
from ops.utils.torch_utils import ModelEMA, smart_optimizer
from ops.utils.torch_utils import one_linear
from utils.postprocess import PostProcess
import torchvision


def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


class DetrModel(LightningModule):

    def __init__(self, hyp):
        super(DetrModel, self).__init__()
        self.hyp = hyp

    def forward(self, x, orig_target_sizes):
        return self(x, orig_target_sizes)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        preds = self(imgs, orig_target_sizes)

        loss_dict = self.criterion(preds, targets)  # box, obj, cls
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = loss

        self.log_dict(loss_dict,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True,
                      prog_bar=True,
                      batch_size=self.trainer.train_dataloader.batch_size)

        # lightning 的 loss / accumulate ，影响收敛
        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema_model.update(self)

    def on_validation_start(self) -> None:
        if not self.trainer.sanity_checking:
            base_ds = get_coco_api_from_dataset(self.val_dataset)
            self.metric = CocoEvaluator(base_ds, ['bbox'])

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        pred, train_out = self.ema_model(imgs, orig_target_sizes)

        loss_dict = self.criterion(train_out, targets)  # box, obj, cls
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = loss

        if not self.trainer.sanity_checking:
            self.log_dict(loss_dict,
                          on_step=True,
                          on_epoch=True,
                          sync_dist=True,
                          prog_bar=True,
                          batch_size=self.trainer.train_dataloader.batch_size)

            res = {target['image_id'].item(): output for target, output in zip(targets, pred)}
            self.metric.update(res)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.metric.synchronize_between_processes()
            self.metric.accumulate()
            self.metric.summarize()

    def configure_model(self) -> None:
        aux_weight_dict = {}
        for i in range(self.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in self.hyp['weight_dict'].items()})
        self.hyp['weight_dict'].update(aux_weight_dict)

        batch_size = self.hyp.batch
        nbs = self.hyp.nbs  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)
        self.hyp['weight_decay'] *= batch_size * accumulate / nbs

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
