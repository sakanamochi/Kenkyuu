"""指定した1モデルと共通の幾何処理からPAF内周楕円を求める。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.images import image_tensor, prepare_image
from core.files import sha
from core.model import TinyUNet
from core.ellipse import Ellipse, EllipseSettings, boundary_ellipse, scale_ellipse


@dataclass(frozen=True)
class Prediction:
    """境界確率、256px座標と元画像座標の楕円、検出状態。"""

    probability: np.ndarray | None
    status: str
    ellipse: Ellipse | None = None
    ellipse_256: Ellipse | None = None


class BoundaryPredictor:
    """checkpoint・計算装置・幾何設定を明示して推論する。"""

    def __init__(self, checkpoint: Path, device: str = "auto",
                 settings: EllipseSettings | None = None) -> None:
        self.settings = settings or EllipseSettings()
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("deviceはauto、cpu、cudaのいずれかにしてください。")
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if selected == "auto" else selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDAを利用できません。deviceにcpuを指定してください。")
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha(self.checkpoint)
        # 実行環境のバージョン型を含むcheckpointも、重み専用の読込で扱う。
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model = TinyUNet()
        # 学習時のコードや監査フォルダを読み込まず、必要な重みだけを使う。
        self.model.load_state_dict(saved["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.epoch = saved.get("epoch")

    def predict(self, image: np.ndarray) -> Prediction:
        gray = prepare_image(image)
        if np.ptp(gray) == 0:
            return Prediction(None, "insufficient_image_evidence")
        with torch.inference_mode():
            logits = self.model(image_tensor(gray, self.device))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        ellipse = boundary_ellipse(probability, self.settings)
        original = scale_ellipse(ellipse, image.shape[0] / 256) if ellipse is not None else None
        return Prediction(probability, "detected" if ellipse is not None else "no_supported_prediction",
                          original, ellipse)
