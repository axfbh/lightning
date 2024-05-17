'''
Created on Feb 20, 2017

@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
# import cv2
import numpy as np
import sys
import os
import shutil
import random

from omegaconf import OmegaConf

from ops.dataset.voc_dataset import VOCDetection


class MyDataSet(VOCDetection):
    def __init__(self, *args, **kwargs):
        super(MyDataSet, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, bboxes, _ = super(MyDataSet, self).__getitem__(item)
        h, w = image.shape[:2]
        normal_bboxes = bboxes / np.array([w, h, w, h], dtype=float)
        return normal_bboxes


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file
        anchors[i][1] *= height_in_cfg_file

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def main():
    args = OmegaConf.load('../config/config.yaml')

    output_dir = '../'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_boxes = []
    for data in MyDataSet(args.train_dataset.path, 'car_train', args):
        all_boxes.append(data)

    annotation_dims = np.concatenate(all_boxes, 0)[:, 2:]

    eps = 0.005
    num_clusters = 6
    width_in_cfg_file = 416
    height_in_cfg_file = 416

    anchor_file = join(output_dir, 'anchors%d.txt' % (num_clusters))
    indices = [random.randrange(annotation_dims.shape[0]) for _ in range(num_clusters)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file)
    print('centroids.shape', centroids.shape)


if __name__ == "__main__":
    main()
