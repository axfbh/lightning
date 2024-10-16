import argparse
from ops.utils.logging import print_args
from ops.models.detection.yolo.model import Yolo


def parse_opt():
    parser = argparse.ArgumentParser()

    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./runs/train/version_40/checkpoints/last.pt',
                        help="resume most recent training")
    parser.add_argument("--cfg", type=str, default="./models/yolo.yaml", help="models.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./data/hyp/hyp-yolo-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch", type=int, default=5, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", type=list, default=[640, 640], help="train, val image size HxW")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    opt = parser.parse_args()
    print_args(vars(opt))

    return opt


def main(opt):
    model = Yolo(opt.cfg)

    model.fit(data=opt.data, imgsz=opt.imgsz, batch=opt.batch, workers=opt.workers)


if __name__ == '__main__':
    main(parse_opt())
