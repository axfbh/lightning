import os.path
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.utilities.rank_zero import rank_zero_info

from ops.utils import extract_ip
from ops.utils.logging import colorstr
from ops.utils.torch_utils import auto_distribute
from ops.utils.callbacks import WarmupLR, LitProgressBar

from ops.dataset.coco_dataset import create_dataloader

from ops.models.detection.detr.base_detr import BaseDetr
from ops.models.detection.detr.deformable_detr import DeformableDETR

from functools import partial


def is_exist_model(path):
    _dir, _file = os.path.split(path)
    prefix, suffix = os.path.splitext(_file)
    name = prefix
    if os.path.exists(path):
        model = OmegaConf.load(path)
    elif os.path.exists(f'./cfg/models/detr/{name}/{name}{suffix}'):
        model = OmegaConf.load(f'./cfg/models/detr/{name}/{name}{suffix}')
    else:
        raise FileNotFoundError(f'{path} file is not exits')
    return model, name


class Detr:
    def __init__(self, model: str, weight: str = None):
        model, name = is_exist_model(model)
        hidden_dim = model.hidden_dim
        num_heads = model.num_heads
        dim_feedforward = model.dim_feedforward
        enc_layers = model.enc_layers
        dec_layers = model.dec_layers
        num_queries = model.num_queries
        num_channels = model.num_channels
        strides = model.strides if hasattr(model, 'strides') else None
        dec_n_points = model.dec_n_points if hasattr(model, 'dec_n_points') else None
        enc_n_points = model.enc_n_points if hasattr(model, 'enc_n_points') else None
        num_classes = model.nc

        self.weight = weight

        self.model = {
            'detr': partial(BaseDetr,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            dim_feedforward=dim_feedforward,
                            enc_layers=enc_layers,
                            dec_layers=dec_layers,
                            num_channels=num_channels,
                            num_queries=num_queries,
                            num_classes=num_classes),
            'deformable_detr': partial(DeformableDETR,
                                       hidden_dim=hidden_dim,
                                       num_heads=num_heads,
                                       dim_feedforward=dim_feedforward,
                                       enc_layers=enc_layers,
                                       dec_layers=dec_layers,
                                       strides=strides,
                                       num_channels=num_channels,
                                       num_queries=num_queries,
                                       num_classes=num_classes,
                                       dec_n_points=dec_n_points,
                                       enc_n_points=enc_n_points)
        }[name]

    def train(self,
              *,
              data: str,
              master_addr: str = extract_ip(),
              master_port: str = "8888",
              node_rank: str = "0",
              num_nodes: int = 1,
              **kwargs):
        """
        @param data: 数据集配置文件的路径.
        @param master_addr: 分布式，主机地址.
        @param master_port: 分布式，主机端口.
        @param node_rank: 分布式，当前机器 id.
        @param num_nodes: 分布式，当前机器 gpu 数量.
        @param imgsz: 定义输入图像的尺寸 HxW.
        @param device: 指定用于训练的计算设备：单个GPU (device=1）、多个 GPU (device=['0','1']）、CPU (device=cpu) 或MPS for Apple silicon (device=mps)
        @param project: 保存训练结果的项目目录名称。允许有组织地存储不同的实验.
        @param name: 训练运行的名称。用于在项目文件夹内创建一个子目录，用于存储训练日志和输出结果.
        @return:
        """
        # ------------ hyp-parameter ------------
        hyp = OmegaConf.load('./cfg/default.yaml')
        hyp.update(kwargs)
        rank_zero_info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

        # ------------ model ------------
        model = self.model(hyp=hyp)

        # ------------ data ------------
        data = OmegaConf.load(data)

        train_dataloader, train_dataset = create_dataloader(data.path,
                                                            hyp.batch,
                                                            hyp,
                                                            image_set='train',
                                                            augment=True,
                                                            workers=hyp.workers,
                                                            shuffle=True,
                                                            persistent_workers=True)

        val_dataloader, val_dataset = create_dataloader(data.path,
                                                        hyp.batch * 2,
                                                        hyp,
                                                        image_set='val',
                                                        augment=False,
                                                        workers=hyp.workers,
                                                        shuffle=False,
                                                        persistent_workers=True)

        # ------------ trainer ------------
        accelerator = hyp.device if hyp.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        warmup_callback = WarmupLR(nbs=hyp.nbs,
                                   momentum=hyp.momentum,
                                   warmup_bias_lr=hyp.warmup_bias_lr,
                                   warmup_epoch=hyp.warmup_epochs,
                                   warmup_momentum=hyp.warmup_momentum)

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='loss',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)
        checkpoint_callback.FILE_EXTENSION = '.pt'

        progress_bar_callback = LitProgressBar(10)

        # plot_callback = PlotLogger(len(bar_val_title))

        self._trainer = L.Trainer(
            accelerator=accelerator,
            devices=hyp.device,
            num_nodes=num_nodes,
            logger=TensorBoardLogger(save_dir=f'./{hyp.project}', name=hyp.name),
            strategy=auto_distribute(num_nodes, hyp.device, master_addr, master_port, node_rank),
            max_epochs=hyp.epochs,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            callbacks=[warmup_callback, checkpoint_callback, progress_bar_callback]
        )

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.weight if hyp.resume else None)

    @property
    def trainer(self):
        return self._trainer
