from ops.utils.callbacks import TQDMProgressBar
from lightning_utilities.core.rank_zero import _info


class ClassifyProgressBar(TQDMProgressBar):

    def get_train_tile(self) -> None:
        _info(
            ("\n" + "%11s" * 4) %
            ("Epoch", "GPU_mem", "lr", "loss")
        )

    def get_val_tile(self) -> str:
        return ("%11s" * 4) % ("F1", "P", "R", 'Accuracy')
