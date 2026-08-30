"""Checkpoint loading and inference for the residual U-Net baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ufill.config import CHECKPOINT_ROOT
from ufill.models.residual_unet import Unet as ResidualUNet


class ResidualUNetPredictor:
    """Load the seven-channel residual U-Net baseline checkpoint."""

    def __init__(self, model_path=None, num_classes=1, backbone="resnet50", cuda=True):
        self.model_path = model_path or CHECKPOINT_ROOT / "residual_unet.pth"
        self.cuda = cuda and torch.cuda.is_available()
        self.net = ResidualUNet(num_classes=num_classes, backbone=backbone)

        device = torch.device("cuda" if self.cuda else "cpu")
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

        print(f"Loaded residual U-Net checkpoint: {self.model_path}")

    def detect_image(self, image):
        image_data = np.expand_dims(image, 0)
        with torch.no_grad():
            inputs = torch.as_tensor(image_data, dtype=torch.float32)
            if self.cuda:
                inputs = inputs.cuda()
            return self.net(inputs)[0].cpu()


Unet = ResidualUNetPredictor

__all__ = ["ResidualUNetPredictor", "Unet"]
