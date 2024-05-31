import numpy as np
import torch
import math

import xml.etree.ElementTree as ET


def batch_images(images, size_divisible=32):
    max_h, max_w = np.array([[img.shape[1], img.shape[2]] for img in images], dtype=int).max(0)

    stride = float(size_divisible)
    max_h = int(math.ceil(float(max_h) / stride) * stride)
    max_w = int(math.ceil(float(max_w) / stride) * stride)

    batched_imgs = torch.zeros((len(images), 3, max_h, max_w), dtype=images[0].dtype)

    for i in range(len(images)):
        img = images[i]
        batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


def batch_labels(labels):
    for i, lb in enumerate(labels):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.cat(labels, 0)


def detect_collate_fn(batch):
    images, labels = zip(*batch)

    batched_imgs = batch_images(images)

    batched_labels = batch_labels(labels)

    return batched_imgs, batched_labels, torch.as_tensor(batched_imgs.shape[-2:])


def voc_bboxes_labels_from_yaml(path, cate=None, name2id=None):
    anno = ET.parse(path).getroot()
    bboxes = []
    classes = []
    for obj in anno.iter("object"):
        if obj.find('difficult').text == '0':
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            name = obj.find("name").text.lower().strip()

            if cate is None or name == cate:
                bboxes.append(box)
                classes.append(name if name2id is None else name2id[name])

    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes, classes
