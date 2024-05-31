import torch
import xml.etree.ElementTree as ET
import os

import numpy as np

from torch.utils.data import Dataset
import ops.cv.io as io
from ops.dataset.utils import voc_bboxes_labels_from_yaml


class VOCDetection(Dataset):

    def __init__(self, root_dir, image_set, image_size, augment, class_name, transform=None):
        super(VOCDetection, self).__init__()

        self.augment = augment
        self._annopath = os.path.join(root_dir, "Annotations", "%s.xml")
        self._imgpath = os.path.join(root_dir, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(root_dir, "ImageSets", "Main", "%s.txt")

        self.id2name = class_name
        self.name2id = dict(zip(class_name.values(), range(len(class_name))))

        self.image_size = image_size

        self.cate = None

        self.transform = transform

        with open(self._imgsetpath % image_set) as f:
            self.img_ids = f.readlines()

        if image_set in ['train', 'trainval', 'val']:
            self.img_ids = [x.strip() for x in self.img_ids]
        else:
            img_ids_flag = [x.strip().split(' ') for x in self.img_ids]
            self.img_ids = [x[0] for x in img_ids_flag if x[1] != '-1']
            self.cate = image_set.split('_')[0]

    def __len__(self):
        return len(self.img_ids) - (2 * 16)

    def __getitem__(self, idx):
        image = io.imread(self._imgpath % self.img_ids[idx])

        bboxes, classes = voc_bboxes_labels_from_yaml(self._annopath % idx, self.cate, self.name2id)

        return image, bboxes, classes
