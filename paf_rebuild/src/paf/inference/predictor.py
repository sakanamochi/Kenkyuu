"""指定した1モデルから境界確率を求める。"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from paf.core.images import image_tensor, prepare_image
from paf.core.model import TinyUNet


@dataclass(frozen=True)
class Prediction:
    """境界確率と、入力が推論対象になったかを表す結果。"""

    probability: np.ndarray | None
    status: str


class BoundaryPredictor:
    """checkpointと計算装置だけを受け取り、境界確率を推定する。"""

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("deviceはauto、cpu、cudaのいずれかにしてください。")
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if selected == "auto" else selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDAを利用できません。deviceにcpuを指定してください。")
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
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
        return Prediction(probability, "boundary_probability")
