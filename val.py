import os
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torchvision.ops.boxes import box_convert
from ops.metric.DetectionMetric import process_batch, ap_per_class

from utils.nms import non_max_suppression
from dataloader import create_dataloader

from ops.utils.logging import print_args, LOGGER
from ops.utils.history_collect import History
from ops.utils.plots import plot_images
from ops.utils.torch_utils import output_to_target

S = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format


@torch.no_grad()
def run(val_loader,
        names,
        model,
        history: History,
        device,
        conf_thres=0.001,
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        plots=True,
        single_cls=False,
        criterion=None):
    model.eval()

    stream = tqdm(val_loader, desc=S, bar_format=TQDM_BAR_FORMAT)

    seen = 0
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    for batch_i, (images, targets, shape) in enumerate(stream):
        images = images.to(device, non_blocking=True) / 255.
        targets = targets.to(device)

        preds, train_out = model(images)

        if criterion:
            image_size = torch.as_tensor(shape, device=device)
            loss += criterion(train_out, targets, image_size)[1]  # box, obj, cls

        preds = non_max_suppression(preds, conf_thres, iou_thres, max_det)

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            labels[:, 0] = labels[:, 0] - 1
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            if single_cls:
                pred[:, 5] = 0

            predn = pred.clone()

            if nl:
                tbox = box_convert(labels[:, 1:5], 'cxcywh', 'xyxy')  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        if plots and batch_i < 3:
            plt_l_image = plot_images(images, targets, names)  # labels
            history.save_image(plt_l_image, mode='label')
            plt_p_image = plot_images(images, output_to_target(preds), names)  # pred
            history.save_image(plt_p_image, mode='pred')

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=20)  # number of targets per class

    LOGGER.info(
        ("%22s" + "%11i" * 2 + "%11.3g" * 4) %
        ("all", seen, nt.sum(), mp, mr, map50, map)
    )

    return mp, mr, map50, map


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--cfg", type=str, default="./models/yolo-v4-v5-l.yaml", help="models.yaml path")
    parser.add_argument("--hyp", type=str, default="./data/hyp/hyp-yolo-v5-low.yaml", help="hyperparameters path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")

    # -------------- 参数值 --------------
    parser.add_argument("--weights", nargs="+", type=str, default="./logs/train/exp/weights/best.pt",
                        help="models path(s)")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--workers", type=int, default=1, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./logs", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")

    return parser.parse_args()


def main():
    opt = parse_opt()
    print_args(vars(opt))

    hyp = OmegaConf.load(Path(opt.hyp))
    cfg = OmegaConf.load(Path(opt.cfg))

    device = opt.device
    model = get_model(cfg)
    model.to(device)
    ckpt = torch.load(opt.weights, map_location="cpu")['ema']  # load
    model.load_state_dict(ckpt.state_dict())

    _, val_loader, names = get_loader(hyp, opt)

    history = History(project_dir=Path(opt.project),
                      name=opt.name,
                      mode='val')

    run(val_loader, names, model, history, device,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        single_cls=opt.single_cls,
        plots=True)


if __name__ == '__main__':
    main()
