import os.path
from pathlib import Path
from typing import List
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_convert
from torchvision.transforms import transforms

from pycocotools import mask as coco_mask

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.utils.torch_utils import nested_tensor_from_tensor_list

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, imgsz: List, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.imgsz = imgsz
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        img = np.array(img)

        sample = A.Compose([
            A.LongestMaxSize(max_size=self.imgsz[0]),
        ], A.BboxParams(format='pascal_voc', label_fields=['classes']))(
            image=img, bboxes=target['boxes'], classes=target['labels']
        )

        if self._transforms is not None:
            sample = self._transforms(sample)

        # albumentations格式 转换成 coco格式
        target = convert_albumen_to_coco_fmt(target, sample)

        img = ToTensorV2()(image=sample['image'])['image'].float() / 255.
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_albumen_to_coco_fmt(target, sample):
    h, w = sample['image'].shape[:-1]

    # x1,y1,x2,y2
    target['boxes'] = box_convert(
        torch.tensor(sample['bboxes'], dtype=torch.float), 'xyxy', 'cxcywh'
    ) / torch.tensor([w, h, w, h])

    target['labels'] = torch.tensor(sample['classes'])
    target['size'] = torch.tensor([h, w])

    return target


def create_dataloader(path,
                      batch,
                      hyp,
                      image_set,
                      augment=False,
                      workers=3,
                      shuffle=False,
                      return_masks=False,
                      persistent_workers=False):
    path = Path(path)

    PATHS = {
        "train": (path / "train2017", path / "annotations" / 'instances_train2017.json'),
        "val": (path / "val2017", path / "annotations" / 'instances_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file,
                            imgsz=hyp.imgsz,
                            transforms=None if augment else None,
                            return_masks=return_masks)

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers

    return DataLoader(dataset=dataset,
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      collate_fn=collate_fn,
                      persistent_workers=persistent_workers), dataset
