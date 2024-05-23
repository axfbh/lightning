from typing import Any

import numpy as np
import torch.nn as nn
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class BasicModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        # self.automatic_optimization = False
