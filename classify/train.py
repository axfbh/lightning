import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import socket

import torch
import torch.distributed

from models.modeling import ResNet
from dataloader import create_dataloader

from ops.utils import extract_ip
from ops.utils.logging import print_args, logger_info_rank_zero_only, colorstr
from ops.utils.callbacks import PlotLogger, WarmupLR

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint
from utils.callbacks import ClassifyProgressBar


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./runs/train1/checkpoints/last.pt',
                        help="resume most recent training")
    # parser.add_argument("--cfg", type=str, default="./models/yolo-v7-l.yaml", help="models.yaml path")
    # parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="../data/hyp/hyp-yolo-v7-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--device", default="gpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"],
                        default="SGD",
                        help="optimizer")
    parser.add_argument("--scheduler", type=str, choices=["Cosine", "MultiStep", "Polynomial", "OneCycleLR"],
                        default="Cosine",
                        help="scheduler")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=3, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./runs", help="save to project/name")
    parser.add_argument("--name", default="train", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    return parser.parse_args()


def setup(opt, hyp):
    # ---------- batch size 参数 ----------
    batch_size = opt.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    tb_logger = TensorBoardLogger(save_dir=opt.project, name=opt.name)

    ddp = DDPStrategy(process_group_backend="nccl" if torch.distributed.is_nccl_available() else 'gloo')

    warmup_callback = WarmupLR(nbs=nbs,
                               momentum=hyp['momentum'],
                               warmup_bias_lr=hyp['warmup_bias_lr'],
                               warmup_epoch=hyp["warmup_epochs"],
                               warmup_momentum=hyp['warmup_momentum'])

    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    bar_callback = ClassifyProgressBar()

    plt_callback = PlotLogger(4)

    checkpoint_callback = ModelCheckpoint(filename='best',
                                          save_last=True,
                                          monitor='fitness',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          enable_version_counter=False)
    checkpoint_callback.FILE_EXTENSION = '.pt'

    trainer = L.Trainer(accelerator=opt.device,
                        devices=1,
                        num_nodes=1,
                        logger=tb_logger,
                        max_epochs=opt.epochs,
                        strategy=ddp,
                        num_sanity_val_steps=1,
                        accumulate_grad_batches=accumulate,
                        log_every_n_steps=1,
                        callbacks=[warmup_callback,
                                   swa_callback,
                                   bar_callback,
                                   plt_callback,
                                   checkpoint_callback])

    print_args(vars(opt))
    logger_info_rank_zero_only(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    return trainer


def main(opt):
    hyp = OmegaConf.load(Path(opt.hyp))
    trainer = setup(opt, hyp)

    model = ResNet(planes=[64, 128, 256, 512],
                   layers=[3, 4, 6, 3],
                   strides=[1, 2, 2, 2],
                   num_classes=10)

    model.hyp = hyp
    model.optim = opt.optimizer
    model.sche = opt.scheduler

    model.save_hyperparameters(dict(vars(opt), **hyp))

    train_loader = create_dataloader('',
                                     opt.image_size,
                                     opt.batch_size,
                                     '',
                                     image_set=True,
                                     augment=True,
                                     local_rank=trainer.local_rank,
                                     rank=trainer.global_rank,
                                     num_nodes=trainer.num_nodes,
                                     workers=opt.workers,
                                     shuffle=True,
                                     persistent_workers=True,
                                     seed=opt.seed)

    val_loader = create_dataloader('',
                                   opt.image_size,
                                   opt.batch_size * 2,
                                   '',
                                   image_set=False,
                                   augment=False,
                                   workers=opt.workers,
                                   shuffle=False,
                                   persistent_workers=True,
                                   seed=opt.seed)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=opt.weights if opt.resume else None)


def init_parallel_process():
    os.environ['MASTER_ADDR'] = extract_ip()
    os.environ['MASTER_PORT'] = '51899'
    os.environ['NODE_RANK'] = '0'


if __name__ == '__main__':
    # init_parallel_process()
    main(parse_opt())
