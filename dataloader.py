import os
import math
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
from ops.augmentations.geometric.transforms import RandomShiftScaleRotate
from ops.augmentations.geometric.resize import ResizeShortLongest
from ops.augmentations.transforms import Mosaic
from ops.utils.logging import colorstr
from ops.utils.torch_utils import torch_distributed_zero_first
import ops.cv.io as io

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


class MyDataSet(VOCDetection):
    def __init__(self, mosaic, aug_mosaic, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mosaic = mosaic
        self.aug_mosaic = aug_mosaic

        self.resize = A.Compose([
            ResizeShortLongest(min_size=self.image_size[0], max_size=self.image_size[1], always_apply=True),
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
            image, bboxes, classes = super().__getitem__(index)
            image_cache[i], bboxes_cache[i], classes_cache[i] = self.resize(image=image, bboxes=bboxes,
                                                                            classes=classes).values()
        return image_cache, bboxes_cache, classes_cache

    def __getitem__(self, item):
        image, bboxes, classes = super().__getitem__(item)
        sample = self.resize(image=image, bboxes=bboxes, classes=classes)

        mosaic = random.random() < self.mosaic and self.augment

        if mosaic:
            image_cache, bboxes_cache, classes_cache = self._update_image_cache()
            sample = self.aug_mosaic(**sample,
                                     image_cache=image_cache,
                                     bboxes_cache=bboxes_cache,
                                     classes_cache=classes_cache)
        sample = self.padding(**sample)

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
    aug_mosaic = A.Compose([
        Mosaic(height=image_size[0] * 2, width=image_size[1] * 2, fill_value=114, always_apply=True),
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
            always_apply=True),
        A.Crop(x_max=image_size[0], y_max=image_size[1], always_apply=True),
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
                        aug_mosaic=aug_mosaic,
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
