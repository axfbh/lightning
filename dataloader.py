import os

import cv2

import torch
from torch.utils.data import DataLoader, distributed
import math
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import transforms

from ops.dataset.voc_dataset import VOCDetection
from ops.dataset.utils import detect_collate_fn
from torchvision.ops.boxes import box_convert
from ops.augmentations.transforms import ResizeShortLongest, RandomShiftScaleRotate, Mosaic
from ops.utils.logging import colorstr
from lightning.fabric.utilities.rank_zero import rank_zero_info
from ops.utils.torch_utils import torch_distributed_zero_first
import ops.cv.io as io
import random

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed


class MyDataSet(VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resize = A.Compose([
            ResizeShortLongest(self.image_size, always_apply=True),
            A.PadIfNeeded(*self.image_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    def __getitem__(self, item):
        image, bboxes, classes = super().__getitem__(item)

        if self.augment:
            arr = np.random.randint(0, len(self.img_ids), 3)
            image_cache = []
            bboxes_cache = []
            classes_cache = []
            for i in arr:
                im, box, cls = super().__getitem__(i)
                image_cache.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                bboxes_cache.append(box)
                classes_cache.append(cls)

            sample = Mosaic(640, 640, False, always_apply=True)(image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                                                bboxes=bboxes,
                                                                image_cache=image_cache,
                                                                bboxes_cache=bboxes_cache)

            if self.augment:
                print(sample['image'].shape)
                io.visualize(sample['image'], sample['bboxes'], [j for i in classes_cache for j in i], self.id2name)

        if self.augment:
            augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)

        sample = self.resize(image=image, bboxes=bboxes, classes=classes)

        if self.augment:
            sample = self.transform(**sample)

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
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HorizontalFlip(p=hyp.fliplr),
        A.VerticalFlip(p=hyp.flipud),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes'], min_area=0.1))

    if augment:
        rank_zero_info(f"{colorstr('albumentations: ')}" + ", ".join(
            f"{x}".replace("always_apply=False, ", "") for x in transform if x.p))

    dataset = MyDataSet(path,
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
