import os
import math
import random

import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops.boxes import box_convert

from lightning.fabric.utilities.rank_zero import rank_zero_info

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.dataset.voc_dataset import voc_image_anno_paths, voc_bboxes_labels_from_xml
from ops.dataset.utils import detect_collate_fn, DataCache
from ops.augmentations.geometric.transforms import RandomShiftScaleRotate
from ops.augmentations.geometric.resize import ResizeShortLongest
from ops.augmentations.transforms import Mosaic
from ops.utils.logging import colorstr
from ops.utils.torch_utils import torch_distributed_zero_first
import ops.cv.io as io

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


class MyDataSet(Dataset):
    def __init__(self,
                 cache,
                 mosaic,
                 aug_mosaic,
                 image_size,
                 augment,
                 transform):

        self.cache = cache
        self.transform = transform
        self.augment = augment
        self.mosaic = mosaic
        self.aug_mosaic = aug_mosaic

        self.resize = A.Compose([
            ResizeShortLongest(min_size=image_size[0], max_size=image_size[1], always_apply=True),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))
        self.padding = A.Compose([
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    def __getitem__(self, item):
        sample = self.cache[item]
        sample = self.resize(**sample)

        mosaic = random.random() < self.mosaic and self.augment
        if mosaic:
            sample = self.aug_mosaic(**sample)

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

    def __len__(self):
        return len(self.cache)


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
    image_paths, anno_paths, cate, name2id = voc_image_anno_paths(path, image_set, names)
    cache = DataCache(image_paths, anno_paths, voc_bboxes_labels_from_xml, cate, name2id)

    aug_mosaic = A.Compose([
        Mosaic(
            reference_data=cache,
            height=image_size[0] * 2,
            width=image_size[1] * 2,
            read_fn=A.Compose([
                ResizeShortLongest(min_size=image_size[0], max_size=image_size[1], always_apply=True),
            ], A.BboxParams(format='pascal_voc', label_fields=['classes'])),
            fill_value=114,
            always_apply=True),
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

    dataset = MyDataSet(
        cache=cache,
        mosaic=hyp['mosaic'],
        aug_mosaic=aug_mosaic,
        image_size=image_size,
        augment=augment,
        transform=transform
    )

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
