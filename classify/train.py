import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from models.modeling import ResNet
from dataloader import create_dataloader

from ops.utils import extract_ip
from ops.utils.logging import print_args, colorstr
from ops.utils.callbacks import WarmupLR
from ops.utils.trainer import Trainer

from lightning.fabric.utilities.rank_zero import rank_zero_info


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
    hyp["weight_decay"] *= batch_size * accumulate / nbs

    warmup_callback = WarmupLR(nbs=nbs,
                               momentum=hyp['momentum'],
                               warmup_bias_lr=hyp['warmup_bias_lr'],
                               warmup_epoch=hyp["warmup_epoch"],
                               warmup_momentum=hyp['warmup_momentum'])

    trainer = Trainer(
        device=opt.device,
        max_epochs=opt.epochs,
        save_dir=opt.project,
        names=opt.name,
        accumulate=accumulate,
        bar_train_title=("loss",),
        bar_val_title=("F1", "P", "R", "Accuracy"),
        callbacks=[warmup_callback]
    )

    print_args(vars(opt))
    rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

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
