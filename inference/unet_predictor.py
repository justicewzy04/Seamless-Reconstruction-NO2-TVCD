"""Checkpoint loading and tensor inference for the temporal U-Net."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ufill.config import CHECKPOINT_ROOT
from ufill.models.temporal_unet import Unet as TemporalUNet


class UNetPredictor:
    """Load a temporal U-Net checkpoint and predict one 55-channel sample."""

    def __init__(self, model_path=None, num_classes=1, backbone="vgg", cuda=True):
        self.model_path = model_path or CHECKPOINT_ROOT / "best_epoch_weights.pth"
        self.cuda = cuda and torch.cuda.is_available()
        self.net = TemporalUNet(num_classes=num_classes, backbone=backbone)

        device = torch.device("cuda" if self.cuda else "cpu")
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

        print(f"Loaded temporal U-Net checkpoint: {self.model_path}")

    def detect_image(self, image):
        """Predict one CHW NumPy array or tensor and return a CPU tensor."""

        image_data = np.expand_dims(image, 0)
        with torch.no_grad():
            inputs = torch.as_tensor(image_data, dtype=torch.float32)
            if self.cuda:
                inputs = inputs.cuda()
            return self.net(inputs)[0].cpu()


Unet = UNetPredictor

__all__ = ["UNetPredictor", "Unet"]
