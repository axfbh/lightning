import os

import cv2

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.fabric.utilities.rank_zero import rank_zero_info

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ops.augmentations.geometric.transforms import RandomShiftScaleRotate
from ops.augmentations.geometric.resize import ResizeShortLongest
from ops.augmentations.transforms import Mosaic
from ops.utils.logging import colorstr
from ops.dataset.voc_dataset import voc_mask_label_from_image, voc_image_mask_paths
from ops.dataset.utils import detect_collate_fn, DataCache

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


class MyDataSet(Dataset):
    def __init__(self,
                 cache,
                 aug_mosaic,
                 image_size,
                 augment,
                 transform):

        self.cache = cache
        self.transform = transform
        self.augment = augment
        self.aug_mosaic = aug_mosaic

        self.resize = A.Compose([
            ResizeShortLongest(min_size=image_size[0], max_size=image_size[1], always_apply=True),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))
        self.padding = A.Compose([
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
                always_apply=True)
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    def __getitem__(self, item):
        sample = self.cache[item]
        sample = self.resize(**sample)

        if self.augment:
            sample = self.aug_mosaic(**sample)
            # io.visualize(sample['image'], sample['bboxes'], sample['classes'])

        sample = self.padding(**sample)

        if self.augment:
            sample = self.transform(**sample)

        image = ToTensorV2()(image=sample['image'])['image'].float()
        mask = torch.FloatTensor(sample['mask'])

        return image, mask

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
    image_paths, mask_paths = voc_image_mask_paths(path, image_set)
    cache = DataCache(image_paths, mask_paths, voc_mask_label_from_image, names)

    aug_mosaic = Mosaic(
        reference_data=cache,
        height=image_size[0],
        width=image_size[0],
        read_fn=A.Compose([
            ResizeShortLongest(min_size=image_size[0], max_size=image_size[1], always_apply=True),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes'])),
        scale=hyp.scale,
        translate=hyp.translate,
        fill_value=114,
        p=hyp['mosaic']
    )

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
