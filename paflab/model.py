from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    """少量CGでも学習しやすい小型U-Net。"""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.encoder_1 = DoubleConv(3, c)
        self.encoder_2 = DoubleConv(c, c * 2)
        self.encoder_3 = DoubleConv(c * 2, c * 4)
        self.bottleneck = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.up_3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.decoder_3 = DoubleConv(c * 8, c * 4)
        self.up_2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.decoder_2 = DoubleConv(c * 4, c * 2)
        self.up_1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.decoder_1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_1 = self.encoder_1(image)
        feature_2 = self.encoder_2(self.pool(feature_1))
        feature_3 = self.encoder_3(self.pool(feature_2))
        bottleneck = self.bottleneck(self.pool(feature_3))
        decoded_3 = self.decoder_3(torch.cat((self.up_3(bottleneck), feature_3), dim=1))
        decoded_2 = self.decoder_2(torch.cat((self.up_2(decoded_3), feature_2), dim=1))
        decoded_1 = self.decoder_1(torch.cat((self.up_1(decoded_2), feature_1), dim=1))
        return self.output(decoded_1)
