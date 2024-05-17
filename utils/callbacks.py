from ops.utils.callbacks import TQDMProgressBar
from ops.utils.logging import LOGGER


class DetectProgressBar(TQDMProgressBar):

    def get_train_tile(self) -> None:
        LOGGER.info(
            ("\n" + "%11s" * 6) %
            ("Epoch", "GPU_mem", "lr", "box_loss", "obj_loss", "cls_loss")
        )

    def get_val_tile(self) -> str:
        return ("%11s" * 6) % ("Images", "Instances", "P", "R", "mAP50", "mAP50-95")
