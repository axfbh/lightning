import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.distributed

from dataloader import create_dataloader

from ops.models.detection import YoloV5, YoloV4, YoloV7
from ops.utils import extract_ip
from ops.utils.logging import print_args, colorstr
from ops.utils.callbacks import PlotLogger, WarmupLR

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import GradientAccumulationScheduler

from utils.callbacks import DetectProgressBar
from lightning.fabric.utilities.rank_zero import rank_zero_info


def parse_opt():
    parser = argparse.ArgumentParser()

    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./runs/train/version_8/checkpoints/last.pt', help="resume most recent training")
    parser.add_argument("--cfg", type=str, default="./models/yolo-v4-v5-n.yaml", help="models.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./data/hyp/hyp-yolo-v5-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=True, help="resume most recent training")
    parser.add_argument("--device", default="gpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"],
                        default="SGD",
                        help="optimizer")
    parser.add_argument("--scheduler", type=str, choices=["Cosine", "MultiStep", "Polynomial", "OneCycleLR"],
                        default="Cosine",
                        help="scheduler")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=2, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./runs", help="save to project/name")
    parser.add_argument("--name", default="train", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--save-period", type=int, default=5, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    return parser.parse_args()


def setup(opt, hyp):
    # ---------- batch size 参数 ----------
    batch_size = opt.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    accumulator_callback = GradientAccumulationScheduler(scheduling={0: 1, (hyp["warmup_epoch"] + 1): accumulate})

    tb_logger = TensorBoardLogger(save_dir=opt.project, name=opt.name)

    # cvs_logger = CSVLogger(save_dir=opt.project, name=opt.name, version=tb_logger.version, flush_logs_every_n_steps=1)

    ddp = DDPStrategy(process_group_backend="nccl" if torch.distributed.is_nccl_available() else 'gloo')

    warmup_callback = WarmupLR(momentum=hyp['momentum'],
                               warmup_bias_lr=hyp['warmup_bias_lr'],
                               warmup_epoch=hyp["warmup_epoch"],
                               warmup_momentum=hyp['warmup_momentum'])

    bar_callback = DetectProgressBar()

    plt_callback = PlotLogger(6)

    checkpoint_callback = ModelCheckpoint(filename='best',
                                          save_last=True,
                                          monitor='fitness_un',
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
                        gradient_clip_val=10.0,
                        # clip gradients' global norm to <=10.0 using gradient_clip_algorithm='norm'
                        gradient_clip_algorithm="norm",
                        num_sanity_val_steps=1,
                        log_every_n_steps=1,
                        callbacks=[
                            accumulator_callback,
                            warmup_callback,
                            bar_callback,
                            plt_callback,
                            checkpoint_callback
                        ])

    print_args(vars(opt))
    rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    return trainer


def main(opt):
    hyp = OmegaConf.load(Path(opt.hyp))
    cfg = OmegaConf.load(Path(opt.cfg))
    data = OmegaConf.load(Path(opt.data))
    trainer = setup(opt, hyp)

    # model = YoloV7(anchors=cfg.anchors, num_classes=cfg.nc, phi='l')
    model = YoloV5(anchors=cfg.anchors,
                   num_classes=cfg.nc,
                   depth_multiple=cfg.depth_multiple,
                   width_multiple=cfg.width_multiple)

    m = model.head  # detection head models
    nl = m.nl  # number of detection layers (to scale hyp)
    nc = m.num_classes
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (max(opt.image_size[0], opt.image_size[1]) / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.hyp = hyp
    model.optim = opt.optimizer
    model.sche = opt.scheduler

    model.save_hyperparameters(dict(vars(opt), **hyp))

    train_loader = create_dataloader(Path(data.train),
                                     opt.image_size,
                                     opt.batch_size,
                                     data.names,
                                     hyp=hyp,
                                     image_set='car_train',
                                     augment=True,
                                     rank=trainer.global_rank,
                                     workers=opt.workers,
                                     shuffle=True,
                                     persistent_workers=True,
                                     seed=opt.seed)

    val_loader = create_dataloader(Path(data.val),
                                   opt.image_size,
                                   opt.batch_size * 2,
                                   data.names,
                                   hyp=hyp,
                                   image_set='car_val',
                                   augment=False,
                                   rank=trainer.global_rank,
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
