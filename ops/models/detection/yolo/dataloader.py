import os
import random

import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops.boxes import box_convert

from lightning.fabric.utilities.rank_zero import rank_zero_info

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.dataset.voc_dataset import voc_image_anno_paths, voc_bboxes_labels_from_xml
from ops.dataset.utils import DataCache
from ops.augmentations.geometric.transforms import RandomShiftScaleRotate
from ops.augmentations.geometric.resize import ResizeShortLongest
from ops.augmentations.transforms import Mosaic
from ops.utils.logging import colorstr

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def yolo_collate_fn(batch):
    im_file, ori_shape, resized_shape, image, target = zip(*batch)

    for i, lb in enumerate(target):
        lb[:, 0] = i  # add target image index for build_targets()

    target = torch.cat(target)

    return {
        'im_file': list(im_file),
        'ori_shape': list(ori_shape),
        'resized_shape': list(resized_shape),
        'img': torch.stack(image),
        'cls': target[:, 1:2],
        'bboxes': target[:, 2:6],
        'batch_idx': target[:, 0]
    }


class YoloDataSet(Dataset):
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
        im_file = self.cache.image_paths[item]
        ori_shape = list(sample['image'].shape[:2])
        sample = self.resize(**sample)

        if self.augment:
            sample = self.aug_mosaic(**sample)
            # io.visualize(sample['image'], sample['bboxes'], sample['classes'])

        sample = self.padding(**sample)

        if self.augment:
            sample = self.transform(**sample)

        image = ToTensorV2()(image=sample['image'])['image'].float()
        bboxes = torch.FloatTensor(sample['bboxes'])
        classes = torch.LongTensor(sample['classes'])[:, None]
        resized_shape = list(image.shape[1:])

        nl = len(bboxes)
        target = torch.zeros((nl, 6))
        if nl:
            box = box_convert(bboxes, 'xyxy', 'cxcywh')
            target[:, 1:2] = classes
            target[:, 2:4] = box[:, :2]
            target[:, 4:6] = box[:, 2:]

        return im_file, ori_shape, resized_shape, image, target

    def __len__(self):
        return len(self.cache)


def create_dataloader(path,
                      batch,
                      names,
                      hyp,
                      image_set=None,
                      augment=False,
                      workers=3,
                      shuffle=False,
                      persistent_workers=False):
    image_paths, anno_paths, cate, name2id = voc_image_anno_paths(path, image_set, names)
    cache = DataCache(image_paths, anno_paths, voc_bboxes_labels_from_xml, cate, name2id)

    aug_mosaic = Mosaic(
        reference_data=cache,
        height=hyp.imgsz[0],
        width=hyp.imgsz[1],
        read_fn=A.Compose([
            ResizeShortLongest(min_size=hyp.imgsz[0], max_size=hyp.imgsz[1], always_apply=True),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes'])),
        scale=hyp.scale,
        translate=hyp.translate,
        value=114,
        p=hyp.mosaic
    )

    transform = A.Compose([
        RandomShiftScaleRotate(
            scale_limit=(1 - hyp.scale, 1 + hyp.scale),
            shift_limit_x=(0.5 - hyp.translate, 0.5 + hyp.translate),
            shift_limit_y=(0.5 - hyp.translate, 0.5 + hyp.translate),
            rotate_limit=(hyp.degrees, hyp.degrees),
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

    dataset = YoloDataSet(
        cache=cache,
        aug_mosaic=aug_mosaic,
        image_size=hyp.imgsz,
        augment=augment,
        transform=transform
    )

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers

    return DataLoader(dataset=dataset,
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      collate_fn=yolo_collate_fn,
                      persistent_workers=persistent_workers)
