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
from ops.augmentations.transforms import ResizeShortLongest, RandomShiftScaleRotate, Mosaic
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

    def _update_image_cache(self):
        indices = random.choices(range(len(self.img_ids)), k=3)  # 3 additional image indices
        random.shuffle(indices)
        image_cache = [None for _ in range(3)]
        bboxes_cache = [None for _ in range(3)]
        classes_cache = [None for _ in range(3)]
        for i, index in enumerate(indices):
            image_cache[i], bboxes_cache[i], classes_cache[i] = super().__getitem__(index)
        return image_cache, bboxes_cache, classes_cache

    def __getitem__(self, item):
        image, bboxes, classes = super().__getitem__(item)
        mosaic = random.random() < self.mosaic and self.augment

        if mosaic:
            image_cache, bboxes_cache, classes_cache = self._update_image_cache()
            sample = self.mosaic_aug(image=image,
                                     bboxes=bboxes,
                                     classes=classes,
                                     image_cache=image_cache,
                                     bboxes_cache=bboxes_cache,
                                     classes_cache=classes_cache)
            io.visualize(cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR),
                         np.array(sample['bboxes']),
                         sample['classes'],
                         self.id2name)
        else:
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
    random_affine = RandomShiftScaleRotate(
        scale_limit=(1 - hyp.scale, 1 + hyp.scale),
        shift_limit_x=(0.5 - hyp.translate, 0.5 + hyp.translate),
        shift_limit_y=(0.5 - hyp.translate, 0.5 + hyp.translate),
        rotate_limit=(0, 0),
        border_mode=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
        position=RandomShiftScaleRotate.PositionType.TOP_LEFT,
        always_apply=True)

    mosaic_aug = A.Compose([
        Mosaic(height=image_size[0] * 2, width=image_size[1] * 2, fill_value=114, always_apply=True),
        random_affine,
        A.Crop(x_max=image_size[0], y_max=image_size[1], always_apply=True),
        A.HueSaturationValue(always_apply=True),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HorizontalFlip(p=hyp.fliplr),
        A.VerticalFlip(p=hyp.flipud),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes'], min_visibility=0.2))

    transform = A.Compose([
        random_affine,
        A.HueSaturationValue(always_apply=True),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HorizontalFlip(p=hyp.fliplr),
        A.VerticalFlip(p=hyp.flipud),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes'], min_visibility=0.2))

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
