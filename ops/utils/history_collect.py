import os
import yaml
from functools import singledispatch
from datetime import datetime
from pathlib import Path
from typing import Dict, Union
from copy import deepcopy

import numpy as np
import cv2

import torch

from ops.utils import threaded
from ops.utils.plots import Annotator
from ops.utils.torch_utils import de_parallel


def yaml_save(file: Union[str, Path] = "data.yaml", data={}):
    # Single-line safe yaml saving
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.is_reset = True
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __iadd__(self, other):
        self.val = other
        self.sum += other
        self.count += 1
        self.avg = self.sum / self.count
        return self

    def __getitem__(self, item):
        return self.avg[item]

    def __repr__(self):
        return str(self.avg)

    def __float__(self):
        return self.avg


class History:
    def __init__(self,
                 project_dir: Path,
                 name: str,
                 mode: str,
                 save_period=-1,
                 best_fitness=None,
                 yaml_args: Dict = None):
        project_dir = project_dir.joinpath(mode)
        # ------------- 根据 时间创建文件夹 ---------------
        if not project_dir.exists():
            project_dir.mkdir()

        i = 0
        while True:
            exp_dir = project_dir.joinpath(name + str(i) if i else name + '')
            if not exp_dir.exists():
                exp_dir.mkdir()
                if mode == 'train':
                    weight_dir = exp_dir.joinpath('weights')
                    weight_dir.mkdir()
                    self.weight_dir = weight_dir
                break
            i += 1

        if yaml_args is not None:
            for k, v in yaml_args.items():
                yaml_save(exp_dir.joinpath(f"{k}.yaml"), v)

        self.exp_dir = exp_dir
        self.best_fitness = best_fitness
        self.save_period = save_period
        self.save_id = 1

    @threaded
    def save(self, model, ema, optimizer, epoch: int, last_iter: int, fitness: float):
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness

        save_dict = {
            'last_epoch': epoch,
            'last_iter': last_iter,
            "best_fitness": fitness,
            'optimizer_name': optimizer.__class__.__name__,
            'optimizer': optimizer.state_dict(),
            'models': deepcopy(de_parallel(model)),
            "ema": deepcopy(ema.ema),
            "updates": ema.updates,
            "date": datetime.now().isoformat(),
        }

        # ---------- save last models --------------
        last_pt_path = self.weight_dir.joinpath('last.pt')
        torch.save(save_dict, last_pt_path)

        # ---------- save best models --------------
        if fitness == self.best_fitness:
            best_pt_path = self.weight_dir.joinpath('best.pt')
            torch.save(save_dict, best_pt_path)

        # ---------- save period models --------------
        if (epoch + 1) % self.save_period == 0:
            weights_pt_path = self.weight_dir.joinpath(f'weights{str(epoch)}.pt')
            torch.save(save_dict, weights_pt_path)

    @threaded
    def save_image(self, image: Union[Annotator, np.ndarray], mode='label'):
        if isinstance(image, Annotator):
            image.save(str(self.exp_dir.joinpath(f"image_{mode}_{str(self.save_id)}.jpg")))
        else:
            cv2.imwrite(str(self.exp_dir.joinpath(f"image_{mode}_{str(self.save_id)}.jpg")), image)
        self.save_id += 1
