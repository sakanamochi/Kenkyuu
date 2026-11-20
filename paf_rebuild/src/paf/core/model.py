"""境界のlogitを出力する3段のTiny U-Net。"""

import torch
from torch import nn


def convolution_block(inputs: int, outputs: int) -> nn.Sequential:
    """畳み込み・BatchNorm・ReLUを2回適用する。"""
    layers = []
    for channels in (inputs, outputs):
        layers.extend([
            nn.Conv2d(channels, outputs, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ])
    return nn.Sequential(*layers)


class TinyUNet(nn.Module):
    """同一輝度3chの画像から、内周境界の1ch logitを求める。"""

    def __init__(self) -> None:
        super().__init__()
        channels = (16, 32, 64)
        previous = 3
        # 保存重みを厳密に読み込めるよう、層名とテンソル形状を固定する。
        for level, width in enumerate(channels, start=1):
            self.add_module(f"encoder_{level}", convolution_block(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = convolution_block(64, 128)
        for level, width in reversed(list(enumerate(channels, start=1))):
            self.add_module(f"up_{level}", nn.ConvTranspose2d(width * 2, width, 2, stride=2))
            self.add_module(f"decoder_{level}", convolution_block(width * 2, width))
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = []
        current = image
        for level in range(1, 4):
            current = self.get_submodule(f"encoder_{level}")(current)
            features.append(current)
            current = self.pool(current)
        current = self.bottleneck(current)
        for level in range(3, 0, -1):
            current = self.get_submodule(f"up_{level}")(current)
            current = torch.cat((current, features[level - 1]), dim=1)
            current = self.get_submodule(f"decoder_{level}")(current)
        return self.output(current)
