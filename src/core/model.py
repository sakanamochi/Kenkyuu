"""PAF内周境界確率を推定するTiny U-Net16（482,737パラメータ）と重み読込。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """3×3畳み込み＋BatchNorm＋ReLUの2回連続ブロック。"""

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
    """縮小・拡大各3段、スキップ接続3本を持つ小型U-Net。"""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = int(base_channels)
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


def count_parameters(model: nn.Module) -> int:
    """学習可能パラメータ数を返す。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def resolve_device(device_name: str = "auto") -> torch.device:
    """指定された実行デバイスを解決する。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できない環境でcudaが指定されました。")
    return device


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    default_base_channels: int = 16,
) -> tuple[TinyUNet, dict[str, Any]]:
    """学習済みチェックポイントを読み込み、推論モードのモデルとメタ情報を返す。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"モデル重みが見つかりません: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("base_channels", default_base_channels))
    model = TinyUNet(base_channels)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
