import os
import math
import numpy as np
import random

import cv2

import torch
from torch.utils.data import DataLoader, distributed
from torchvision.ops.boxes import box_convert

from lightning.fabric.utilities.rank_zero import rank_zero_info

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.dataset.voc_dataset import VOCDetection
from ops.dataset.utils import detect_collate_fn
from ops.augmentations.transforms import ResizeShortLongest, RandomShiftScaleRotate
from ops.utils.logging import colorstr
from utils.utils import box_candidates
from ops.utils.torch_utils import torch_distributed_zero_first
import ops.cv.io as io

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


# def seed_worker(worker_id):
#     # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
#
# def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
#     # HSV color-space augmentation
#     if hgain or sgain or vgain:
#         r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
#         hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
#         dtype = im.dtype  # uint8
#
#         x = np.arange(0, 256, dtype=r.dtype)
#         lut_hue = ((x * r[0]) % 180).astype(dtype)
#         lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#         lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
#
#         im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
#         cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed


class MyDataSet(VOCDetection):
    def __init__(self, mosaic, mosaic_aug, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mosaic = mosaic
        self.mosaic_aug = mosaic_aug

        self.resize = A.Compose([
            ResizeShortLongest(self.image_size, always_apply=True),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))
        self.padding = A.Compose([
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    def load_mosaic(self, item):
        bboxes4, classes4, segments4 = [], [], []
        indices = [item] + random.choices(range(len(self.img_ids)), k=3)  # 3 additional image indices
        random.shuffle(indices)
        height, width = self.image_size
        mosaic_border = [-height // 2, -width // 2]

        yc, xc = (int(random.uniform(-x, 2 * y + x)) for x, y in zip(mosaic_border, [height, width]))
        img4 = np.full((height * 2, width * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        for i, index in enumerate(indices):
            image, bboxes, classes = super().__getitem__(index)
            sample = self.resize(image=image, bboxes=bboxes, classes=classes)
            image, bboxes, classes = sample['image'], np.array(sample['bboxes']), np.array(sample['classes'], dtype=int)
            nt = len(bboxes)
            (h, w) = image.shape[:2]

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, width * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(height * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, width * 2), min(height * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            if nt:
                prev_bboxes = bboxes.copy()
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] + padw, 0, 2 * width)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] + padh, 0, 2 * height)
                j = box_candidates(prev_bboxes.T, bboxes.T, area_thr=0.01)
                bboxes, classes = bboxes[j], classes[j]

                bboxes4.append(bboxes)
                classes4.extend(classes.tolist())

        bboxes4 = np.concatenate(bboxes4, 0)
        sample = self.mosaic_aug(image=img4, bboxes=bboxes4, classes=classes4)
        return sample

    def __getitem__(self, item):
        if random.random() < self.mosaic and self.augment:
            sample = self.load_mosaic(item)
            # io.visualize(cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR),
            #              sample['bboxes'],
            #              sample['classes'],
            #              self.id2name)
        else:
            image, bboxes, classes = super().__getitem__(item)
            sample = self.resize(image=image, bboxes=bboxes, classes=classes)
            sample = self.padding(**sample)

        if self.augment:
            sample = self.transform(**sample)
            # io.visualize(sample['image'], sample['bboxes'], sample['classes'], self.id2name)

        image = ToTensorV2()(image=sample['image'])['image'].float()
        bboxes = torch.FloatTensor(sample['bboxes'])
        classes = torch.LongTensor(sample['classes'])[:, None]

        nl = len(bboxes)
        target = torch.zeros((nl, 6))
        if nl:
            box = box_convert(bboxes, 'xyxy', 'cxcywh')
            target[:, 1:2] = classes
            target[:, 2:4] = box[:, :2]
            target[:, 4:6] = box[:, 2:]

        return image, target


def create_dataloader(path,
                      image_size,
                      batch_size,
                      names,
                      image_set=None,
                      hyp=None,
                      augment=False,
                      workers=3,
                      shuffle=False,
                      persistent_workers=False):
    mosaic_aug = A.Compose([
        RandomShiftScaleRotate(
            scale_limit=(1 - hyp.scale, 1 + hyp.scale),
            shift_limit_x=(0.5 - hyp.translate, 0.5 + hyp.translate),
            shift_limit_y=(0.5 - hyp.translate, 0.5 + hyp.translate),
            rotate_limit=(0, 0),
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            position=RandomShiftScaleRotate.PositionType.TOP_LEFT,
            always_apply=True),
        A.Crop(x_max=640, y_max=640, always_apply=True)
    ], A.BboxParams(format='pascal_voc', label_fields=['classes'], min_visibility=0.2))

    transform = A.Compose([
        RandomShiftScaleRotate(
            scale_limit=(1 - hyp.scale, 1 + hyp.scale),
            shift_limit_x=(0.5 - hyp.translate, 0.5 + hyp.translate),
            shift_limit_y=(0.5 - hyp.translate, 0.5 + hyp.translate),
            rotate_limit=(0, 0),
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            position=RandomShiftScaleRotate.PositionType.TOP_LEFT,
            always_apply=True
        ),
        A.HueSaturationValue(always_apply=True),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HorizontalFlip(p=hyp.fliplr),
        A.VerticalFlip(p=hyp.flipud),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes'], min_visibility=0.1))

    if augment:
        rank_zero_info(f"{colorstr('albumentations: ')}" + ", ".join(
            f"{x}".replace("always_apply=False, ", "") for x in transform if x.p))

    dataset = MyDataSet(mosaic=hyp['mosaic'],
                        mosaic_aug=mosaic_aug,
                        root_dir=path,
                        image_set=image_set,
                        image_size=image_size,
                        class_name=names,
                        augment=augment,
                        transform=transform)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      collate_fn=detect_collate_fn,
                      persistent_workers=persistent_workers)
