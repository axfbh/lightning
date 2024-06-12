import os
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, distributed

import numpy as np

from torchvision import transforms, datasets

import random

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path,
                      image_size,
                      batch_size,
                      augment=False,
                      rank=0,
                      workers=3,
                      shuffle=False,
                      persistent_workers=False):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    if image_size:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    dataset = datasets.CIFAR100(r"D:\dataset\cifa100",
                               train=augment,
                               transform=transform,
                               download=True)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # generator = torch.Generator()
    # generator.manual_seed(6148914691236517205 + seed + rank)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      # collate_fn=detect_collate_fn,
                      # worker_init_fn=seed_worker,
                      persistent_workers=persistent_workers,
                      # generator=generator
                      )
