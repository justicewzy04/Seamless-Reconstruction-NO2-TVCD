"""ConvLSTM network for seven-frame satellite gap-filling regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """A convolutional LSTM cell that preserves spatial structure."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state(self, x):
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        hidden = x.new_zeros(shape)
        cell = x.new_zeros(shape)
        return hidden, cell

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state(x)
        hidden, cell = state
        gates = self.gates(torch.cat((x, hidden), dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride,
                padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvLSTM(nn.Module):
    """Encoder-ConvLSTM-decoder model with the existing BCHW interface.

    The seven input channels are interpreted as seven ordered single-channel
    frames. Each frame is encoded with shared weights, processed by a stacked
    ConvLSTM, and the final recurrent state is decoded into a dense map.
    """

    def __init__(
        self,
        num_classes=1,
        in_channels=7,
        hidden_channels=64,
        backbone="convlstm",
        pretrained=False,
    ):
        super().__init__()
        if backbone != "convlstm":
            raise ValueError("backbone must be 'convlstm'")
        if pretrained:
            raise ValueError(
                "Pretrained ConvLSTM weights are not bundled; load a checkpoint instead"
            )
        if in_channels < 1:
            raise ValueError("in_channels must be positive")

        self.in_channels = in_channels
        self.backbone = backbone
        self.frame_encoder = nn.Sequential(
            ConvBlock(1, 16, stride=2),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, hidden_channels, stride=2),
        )
        self.recurrent1 = ConvLSTMCell(hidden_channels, hidden_channels)
        self.recurrent2 = ConvLSTMCell(hidden_channels, hidden_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, channels, height, width]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        input_size = x.shape[-2:]
        state1 = None
        state2 = None
        for timestep in range(self.in_channels):
            frame = x[:, timestep:timestep + 1]
            encoded = self.frame_encoder(frame)
            hidden1, cell1 = self.recurrent1(encoded, state1)
            state1 = hidden1, cell1
            hidden2, cell2 = self.recurrent2(hidden1, state2)
            state2 = hidden2, cell2

        output = self.decoder(state2[0])
        if output.shape[-2:] != input_size:
            output = F.interpolate(
                output, size=input_size, mode="bilinear", align_corners=False
            )
        return output

    def freeze_backbone(self):
        for module in (self.frame_encoder, self.recurrent1, self.recurrent2):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_backbone(self):
        for module in (self.frame_encoder, self.recurrent1, self.recurrent2):
            for parameter in module.parameters():
                parameter.requires_grad = True

