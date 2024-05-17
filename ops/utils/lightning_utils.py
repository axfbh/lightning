# import os
# from pathlib import Path
# from typing import Union, Any, TYPE_CHECKING, Dict
#
# from lightning.pytorch import loggers as pl_loggers
# from lightning_utilities.core.imports import RequirementCache
#
# _TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")
# _TENSORBOARDX_AVAILABLE = RequirementCache("tensorboardX")
# if TYPE_CHECKING:
#     # assumes at least one will be installed when type checking
#     if _TENSORBOARD_AVAILABLE:
#         from torch.utils.tensorboard import SummaryWriter
#     else:
#         from tensorboardX import SummaryWriter  # type: ignore[no-redef]
#
#
# class TensorBoardLogger(pl_loggers.TensorBoardLogger):
#
#     def __init__(
#             self,
#             save_dir,
#             name="lightning_logs",
#             version=None,
#             log_graph: bool = False,
#             default_hp_metric: bool = True,
#             prefix: str = "",
#             sub_dir=None,
#             **kwargs: Any,
#     ):
#         check_dir = Path(os.path.join(save_dir, name))
#
#         i = 0
#         while True:
#             ver_dir = check_dir.joinpath(version + str(i) if i else version + '')
#             if not ver_dir.exists():
#                 version += str(i) if i else ''
#                 break
#             i += 1
#
#         super().__init__(
#             save_dir=save_dir,
#             name=name,
#             version=version,
#             log_graph=log_graph,
#             default_hp_metric=default_hp_metric,
#             prefix=prefix,
#             sub_dir=sub_dir,
#             **kwargs,
#         )
