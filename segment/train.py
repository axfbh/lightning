import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from dataloader import create_dataloader

from ops.utils.logging import print_args, colorstr
from ops.utils.callbacks import WarmupLR
from ops.utils.trainer import Trainer
# from ops.models.segmentation.unet import Unet
from models.modeling import Unet
from ops.models.segmentation.liteseg import LiteSeg

from lightning.fabric.utilities.rank_zero import rank_zero_info


def parse_opt():
    parser = argparse.ArgumentParser()

    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./runs/train/version_15/checkpoints/last.pt',
                        help="resume most recent training")
    # parser.add_argument("--cfg", type=str, default="./models/yolo.yaml", help="models.yaml path")
    parser.add_argument("--data", type=str, default="../data/voc-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="../data/hyp/hyp-yolo-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[512, 512], help="train, val image size HxW")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--device", default="gpu", help="cpu, gpu, tpu, ipu, hpu, mps, auto")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"],
                        default="SGD",
                        help="optimizer")
    parser.add_argument("--scheduler", type=str, choices=["Cosine", "MultiStep", "Polynomial", "OneLinearLR"],
                        default="OneLinearLR",
                        help="scheduler")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=2, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./runs", help="save to project/name")
    parser.add_argument("--name", default="train", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
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
        max_epochs=opt.epochs,
        save_dir=opt.project,
        names=opt.name,
        device=opt.device,
        nproc_per_node=1,
        accumulate=accumulate,
        bar_train_title=("cnt_loss",),
        bar_val_title=("Acc", "Acc_class", "mIoU", "fwIoU"),
        callbacks=[warmup_callback]
    )

    print_args(vars(opt))
    rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    return trainer


def main(opt):
    hyp = OmegaConf.load(Path(opt.hyp))
    # cfg = OmegaConf.load(Path(opt.cfg))
    data = OmegaConf.load(Path(opt.data))
    trainer = setup(opt, hyp)

    model = LiteSeg(num_classes=21)
    model.hyp = hyp
    model.opt = opt

    model.save_hyperparameters(dict(vars(opt), **hyp))

    train_loader = create_dataloader(Path(data.train),
                                     opt.image_size,
                                     opt.batch_size,
                                     data.colors,
                                     hyp=hyp,
                                     image_set='train',
                                     augment=True,
                                     workers=opt.workers,
                                     shuffle=True,
                                     persistent_workers=True)

    val_loader = create_dataloader(Path(data.val),
                                   opt.image_size,
                                   opt.batch_size * 2,
                                   data.colors,
                                   hyp=hyp,
                                   image_set='val',
                                   augment=False,
                                   workers=opt.workers,
                                   shuffle=False,
                                   persistent_workers=True)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=opt.weights if opt.resume else None)


if __name__ == '__main__':
    main(parse_opt())
