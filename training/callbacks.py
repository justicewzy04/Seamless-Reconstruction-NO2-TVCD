"""Training-history logging."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import scipy.signal
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class LossHistory:
    """Persist scalar losses, TensorBoard events, and a loss curve."""

    def __init__(self, log_dir, val_loss_flag=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.val_loss_flag = val_loss_flag
        self.losses = []
        self.val_loss = []
        self.writer = SummaryWriter(str(self.log_dir))

    def append_loss(self, epoch, loss, val_loss=None):
        self.losses.append(loss)
        self._append_value("epoch_loss.txt", loss)
        self.writer.add_scalar("loss", loss, epoch)

        if self.val_loss_flag:
            self.val_loss.append(val_loss)
            self._append_value("epoch_val_loss.txt", val_loss)
            self.writer.add_scalar("val_loss", val_loss, epoch)

        self.loss_plot()

    def _append_value(self, filename, value):
        with (self.log_dir / filename).open("a", encoding="utf-8") as file:
            file.write(f"{value}\n")

    def loss_plot(self):
        epochs = range(len(self.losses))
        plt.figure()
        plt.plot(epochs, self.losses, "red", linewidth=2, label="train loss")
        if self.val_loss_flag:
            plt.plot(epochs, self.val_loss, "coral", linewidth=2, label="validation loss")

        window = 5 if len(self.losses) < 25 else 15
        if len(self.losses) >= window:
            plt.plot(
                epochs,
                scipy.signal.savgol_filter(self.losses, window, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smoothed train loss",
            )

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(self.log_dir / "epoch_loss.png")
        plt.close()

    def close(self):
        self.writer.close()


__all__ = ["LossHistory"]
