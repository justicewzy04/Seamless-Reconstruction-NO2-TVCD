"""Seven-channel view of the full geospatial dataset for baseline models."""

from __future__ import annotations

import torch

from .geo_dataset import APDataset


class BaselineDataset(APDataset):
    """Return four GEMS, GEOS-CF, previous-day and next-day TROPOMI channels.

    The full temporal model consumes 55 channels. ConvLSTM, SegFormer and the
    residual U-Net baselines were defined for the seven channels selected here.
    """

    CHANNELS = (0, 1, 2, 3, 4, 21, 38)

    def __getitem__(self, index):
        inputs, target = super().__getitem__(index)
        if inputs.shape[0] <= self.CHANNELS[-1]:
            raise ValueError(
                f"Expected at least {self.CHANNELS[-1] + 1} input channels, "
                f"got {inputs.shape[0]}"
            )
        channel_index = torch.tensor(self.CHANNELS, device=inputs.device)
        return inputs.index_select(0, channel_index), target
