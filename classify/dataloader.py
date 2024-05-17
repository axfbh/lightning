import os
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, distributed

import numpy as np

from torchvision import transforms, datasets

from ops.dataset.utils import detect_collate_fn
from ops.utils.logging import LOGGER, colorstr
from ops.utils.torch_utils import torch_distributed_zero_first
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path,
                      image_size,
                      batch_size,
                      names,
                      image_set=None,
                      hyp=None,
                      augment=False,
                      local_rank=0,
                      rank=0,
                      num_nodes=1,
                      workers=3,
                      shuffle=False,
                      persistent_workers=False,
                      seed=0):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    # transform = A.Compose([
    #     A.Normalize(),
    #     ToTensorV2(),
    # ])

    # if augment:
    #     LOGGER.info(f"{colorstr('albumentations: ')}" + ", ".join(
    #         f"{x}".replace("always_apply=False, ", "") for x in transform if x.p))

    # with torch_distributed_zero_first(local_rank, num_nodes):  # init dataset *.cache only once if DDP
    dataset = datasets.CIFAR10(r"D:\Desktop\backbone_exp\data\cifar-10-batches-py",
                               train=image_set,
                               transform=transform,
                               download=True)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + rank)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=nw,
                      pin_memory=PIN_MEMORY,
                      # collate_fn=detect_collate_fn,
                      worker_init_fn=seed_worker,
                      persistent_workers=persistent_workers,
                      generator=generator)
