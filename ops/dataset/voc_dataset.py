import os
from typing import Dict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import ops.cv.io as io


def voc_image_anno_paths(root_dir, image_set, id2name: Dict):
    _annopath = os.path.join(root_dir, "Annotations", "%s.xml")
    _imgpath = os.path.join(root_dir, "JPEGImages", "%s.jpg")
    _imgsetpath = os.path.join(root_dir, "ImageSets", "Main", "%s.txt")

    name2id = dict(zip(id2name.values(), range(len(id2name))))

    cate = None

    with open(_imgsetpath % image_set) as f:
        img_ids = f.readlines()

    if image_set in ['train', 'trainval', 'val']:
        image_path = [_imgpath % x.strip() for x in img_ids]
        anno_path = [_annopath % x.strip() for x in img_ids]
    else:
        img_ids_flag = [x.strip().split(' ') for x in img_ids]
        image_path = [_imgpath % x[0] for x in img_ids_flag if x[1] != '-1']
        anno_path = [_annopath % x[0] for x in img_ids_flag if x[1] != '-1']
        cate = image_set.split('_')[0]

    return image_path, anno_path, cate, name2id


def voc_image_mask_paths(root_dir, image_set):
    _annopath = os.path.join(root_dir, "SegmentationClass", "%s.png")
    _imgpath = os.path.join(root_dir, "JPEGImages", "%s.jpg")
    _imgsetpath = os.path.join(root_dir, "ImageSets", "Segmentation", "%s.txt")

    with open(_imgsetpath % image_set) as f:
        img_ids = f.readlines()

    if image_set in ['train', 'trainval', 'val']:
        image_path = [_imgpath % x.strip() for x in img_ids]
        anno_path = [_annopath % x.strip() for x in img_ids]
    else:
        raise ValueError("image_set must in ['train', 'trainval', 'val'] ")

    return image_path, anno_path


def voc_bboxes_labels_from_xml(path, cate: str = None, name2id: Dict = None) -> Dict:
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
    return {"bboxes": bboxes, "classes": classes}


def voc_mask_label_from_image(path, name2color: Dict) -> Dict:
    gt_mask = io.imread(path)
    h, w = gt_mask.shape[:-1]
    masks = np.zeros((20, h, w), dtype=np.uint8)
    colors = list(name2color.values())[1:]
    for i, color in enumerate(colors):
        masks[i, (gt_mask == color).all(-1)] = 255
    return {"mask": masks}
