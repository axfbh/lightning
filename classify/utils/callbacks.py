from ops.utils.callbacks import TQDMProgressBar
from ops.utils.logging import LOGGER


class ClassifyProgressBar(TQDMProgressBar):

    def get_train_tile(self) -> None:
        LOGGER.info(
            ("\n" + "%11s" * 4) %
            ("Epoch", "GPU_mem", "lr", "loss")
        )

    def get_val_tile(self) -> str:
        return ("%11s" * 4) % ("F1", "P", "R", 'Accuracy')
