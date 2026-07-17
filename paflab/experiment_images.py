from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from analysis.ellipse_baseline import evaluate_ellipses, preprocess_image
from analysis.ellipse_ransac import (
    _point_distances,
    fit_contour_ransac_candidates,
    select_paf_inner_candidate,
)
from analysis.zhang_arc_detector import (
    detect_zhang_arc_candidates,
    draw_zhang_arcs,
    select_zhang_inner_boundary,
)
from paflab.camera_effects import EFFECTS
from paflab.evaluate_cnn import probability_ellipse
from paflab.image_io import imread, imwrite
from paflab.labels import fit_label_ellipse, scale_ellipse
from paflab.model import TinyUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EFFECT_LABELS = {
    "clean": "劣化なし",
    "black_rectangle": "単純黒矩形",
    "sensor_whiteout": "センサ白飛びproxy",
    "sensor_black_crush": "センサ黒つぶれproxy",
}

METHOD_LABELS = {
    "canny_ransac_inner_pair": "Canny + 輪郭別RANSAC",
    "zhang2019_arc_reproduction": "Zhang 2019型（再現実装）",
    "cnn_ransac": "CNN＋weighted RANSAC",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(dataset_dir: str | Path) -> tuple[Path, dict, dict[str, dict]]:
    """GUI用にmanifestとsample_id索引をUTF-8で読み込む。"""
    dataset = _project_path(dataset_dir)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    return dataset, manifest, samples


def _ellipse_to_dict(ellipse) -> dict | None:
    if ellipse is None:
        return None
    return {
        "center_x": float(ellipse[0][0]),
        "center_y": float(ellipse[0][1]),
        "axis_1": float(ellipse[1][0]),
        "axis_2": float(ellipse[1][1]),
        "angle_deg": float(ellipse[2]),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _draw_dashed_ellipse(
    image: np.ndarray,
    ellipse,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    center = tuple(int(round(value)) for value in ellipse[0])
    axes = tuple(max(1, int(round(value / 2))) for value in ellipse[1])
    points = cv2.ellipse2Poly(
        center,
        axes,
        int(round(ellipse[2])),
        0,
        360,
        5,
    )
    for index in range(0, len(points) - 1, 2):
        cv2.line(
            image,
            tuple(points[index]),
            tuple(points[index + 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_result(
    image: np.ndarray,
    ground_truth,
    predicted,
    evaluation: dict | None,
) -> np.ndarray:
    """水色破線を正解、成功を緑、失敗を赤として重畳する。"""
    output = image.copy()
    if ground_truth is not None:
        _draw_dashed_ellipse(output, ground_truth, (214, 166, 0), 2)
    if predicted is not None:
        success = bool(evaluation and evaluation["ellipse_iou"] >= 0.8)
        cv2.ellipse(
            output,
            predicted,
            (102, 163, 33) if success else (77, 72, 229),
            3,
            cv2.LINE_AA,
        )
    return output


def _draw_contours(image: np.ndarray, contours) -> np.ndarray:
    output = image.copy()
    palette = (
        (235, 99, 37),
        (11, 158, 245),
        (102, 163, 33),
        (247, 85, 168),
        (153, 72, 236),
    )
    for index, contour in enumerate(contours):
        cv2.drawContours(
            output,
            [contour],
            -1,
            palette[index % len(palette)],
            2,
            cv2.LINE_AA,
        )
    return output


def _draw_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    output = image.copy()
    for rank, candidate in enumerate(candidates[:12], start=1):
        cv2.ellipse(
            output,
            candidate["ellipse"],
            (0, 0, 255) if rank == 1 else (0, 180, 255),
            3 if rank == 1 else 1,
            cv2.LINE_AA,
        )
    return output


def _probability_heatmap(probability: np.ndarray) -> np.ndarray:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_MAGMA)


def _threshold_points(probability: np.ndarray, threshold: float) -> np.ndarray:
    mask = probability >= threshold
    output = np.zeros((*probability.shape, 3), dtype=np.uint8)
    output[mask] = (255, 240, 235)
    return output


def _ransac_inliers(
    probability: np.ndarray,
    threshold: float,
    predicted,
    distance_threshold: float,
) -> np.ndarray:
    rows, columns = np.nonzero(probability >= threshold)
    points = np.column_stack((columns, rows)).astype(np.float32)
    output = np.zeros((*probability.shape, 3), dtype=np.uint8)
    output[rows, columns] = (150, 135, 125)
    if predicted is not None and len(points):
        inliers = _point_distances(points, predicted) <= distance_threshold
        selected = points[inliers].astype(int)
        output[selected[:, 1], selected[:, 0]] = (140, 220, 33)
        cv2.ellipse(output, predicted, (255, 140, 70), 2, cv2.LINE_AA)
    return output


def _setup_japanese_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def _save_overview(stages: list[dict], path: Path) -> None:
    """全中間画像を一覧できる確認用コンタクトシートを保存する。"""
    _setup_japanese_font()
    columns = 4
    rows = math.ceil(len(stages) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.0, rows * 4.0))
    axes = np.asarray(axes).reshape(-1)
    for axis, stage in zip(axes, stages):
        image = stage["image"]
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(stage["label"], fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(stages) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class ExperimentImageGenerator:
    """1枚の入力から劣化・古典法・CNNの中間画像一式を生成する。"""

    def __init__(
        self,
        *,
        cnn_config: str | Path = "config/research_experiment.json",
        baseline_config: str | Path = "config/baseline.json",
        checkpoint: str | Path | None = None,
    ) -> None:
        self.cnn_config_path = _project_path(cnn_config)
        self.baseline_config_path = _project_path(baseline_config)
        self.cnn_config = json.loads(
            self.cnn_config_path.read_text(encoding="utf-8")
        )
        self.baseline = json.loads(
            self.baseline_config_path.read_text(encoding="utf-8")
        )
        self.checkpoint_path = (
            _project_path(checkpoint)
            if checkpoint is not None
            else _project_path(self.cnn_config["artifacts_dir"]) / "cnn_best.pt"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TinyUNet | None = None

    def _load_model(self) -> TinyUNet:
        if self.model is None:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            model = TinyUNet(int(checkpoint["base_channels"]))
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            self.model = model
        return self.model

    def _apply_effect(
        self,
        image: np.ndarray,
        sample: dict,
        ellipse,
        effect: str,
        severity: float,
        seed: int,
    ) -> tuple[np.ndarray, dict]:
        if effect == "clean" or severity == 0.0:
            return image.copy(), {"severity": 0.0, "definition": "未劣化"}
        if effect not in EFFECTS:
            raise ValueError(f"未対応の撮像効果です: {effect}")
        rng = np.random.default_rng(seed)
        if effect == "black_rectangle":
            return EFFECTS[effect](image, ellipse, severity, rng=rng)
        return EFFECTS[effect](image, severity, rng=rng)

    def generate(
        self,
        dataset_dir: str | Path,
        sample_id: str,
        *,
        effect: str,
        severity: float,
        seed: int,
        output_dir: str | Path,
        selected_methods: list[str] | tuple[str, ...] | None = None,
    ) -> dict:
        methods = list(
            METHOD_LABELS if selected_methods is None else selected_methods
        )
        unknown_methods = set(methods) - set(METHOD_LABELS)
        if unknown_methods:
            raise ValueError(
                f"未対応の検出手法です: {', '.join(sorted(unknown_methods))}"
            )
        if not methods:
            raise ValueError("検出手法を1つ以上選択してください")

        dataset, _, samples = load_manifest(dataset_dir)
        if sample_id not in samples:
            raise KeyError(f"sample_idがmanifestにありません: {sample_id}")
        sample = samples[sample_id]
        image = imread(dataset / sample["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(dataset / sample["image"])
        label = json.loads(
            (dataset / sample["label"]).read_text(encoding="utf-8")
        )
        ground_truth = fit_label_ellipse(label)
        severity = float(np.clip(severity, 0.0, 1.0))
        degraded, effect_metadata = self._apply_effect(
            image,
            sample,
            ground_truth,
            effect,
            severity,
            seed,
        )

        stages = [
            {"key": "00-original", "label": "元画像", "image": image},
            {
                "key": "01-degraded",
                "label": f"{EFFECT_LABELS[effect]} {severity:.0%}",
                "image": degraded,
            },
        ]
        method_metadata = {}

        if "canny_ransac_inner_pair" in methods:
            # 輪郭別RANSACの候補から、同心二重楕円なら小さい内周側を選ぶ。
            canny_stages = preprocess_image(degraded, self.baseline["detector"])
            canny_candidates = fit_contour_ransac_candidates(
                canny_stages["contours"],
                degraded.shape,
                self.baseline["ransac"],
                random_seed=seed,
            )
            canny_selected = select_paf_inner_candidate(
                canny_candidates,
                self.baseline["inner_pair_selector"],
            )
            canny_predicted = (
                canny_selected["ellipse"] if canny_selected is not None else None
            )
            canny_evaluation = (
                evaluate_ellipses(canny_predicted, ground_truth, degraded.shape)
                if canny_predicted is not None
                else None
            )
            stages.extend([
                {
                "key": "02-gray",
                "label": "グレースケール",
                "image": canny_stages["gray"],
                },
                {
                "key": "03-gaussian-blur",
                "label": "Gaussian平滑化",
                "image": canny_stages["blurred"],
                },
                {
                "key": "04-canny-edges",
                "label": "Cannyエッジ",
                "image": canny_stages["edges"],
                },
                {
                "key": "05-canny-contours",
                "label": "輪郭分離",
                "image": _draw_contours(degraded, canny_stages["contours"]),
                },
                {
                "key": "06-canny-ransac-candidates",
                "label": "Canny + 輪郭別RANSAC候補",
                "image": _draw_candidates(degraded, canny_candidates),
                },
                {
                "key": "07-canny-result",
                "label": "Canny + 輪郭別RANSAC結果",
                "image": _draw_result(
                    degraded,
                    ground_truth,
                    canny_predicted,
                    canny_evaluation,
                ),
                },
            ])
            method_metadata["canny_ransac_inner_pair"] = {
                "label": METHOD_LABELS["canny_ransac_inner_pair"],
                "candidate_count": len(canny_candidates),
                "detected": _ellipse_to_dict(canny_predicted),
                "evaluation": canny_evaluation,
                "selection_mode": (
                    canny_selected.get("selection_mode")
                    if canny_selected is not None
                    else None
                ),
                "inner_pair": (
                    canny_selected.get("inner_pair")
                    if canny_selected is not None
                    else None
                ),
            }

        if "zhang2019_arc_reproduction" in methods:
            # Zhang 2019型（再現実装）の弧抽出・統合候補。
            zhang_candidates, zhang_stages = detect_zhang_arc_candidates(
                degraded,
                self.baseline["detector"],
                self.baseline["zhang2019_arc"],
            )
            zhang_selected = (
                select_zhang_inner_boundary(
                    zhang_candidates,
                    self.baseline["zhang2019_inner_selector"],
                )
                if zhang_candidates
                else None
            )
            zhang_predicted = (
                zhang_selected["ellipse"] if zhang_selected is not None else None
            )
            zhang_evaluation = (
                evaluate_ellipses(zhang_predicted, ground_truth, degraded.shape)
                if zhang_predicted is not None
                else None
            )
            stages.extend([
                {
                "key": "08-zhang-arcs-candidates",
                "label": "Zhang 2019型（再現実装）の弧・候補",
                "image": draw_zhang_arcs(
                    degraded,
                    zhang_stages,
                    zhang_candidates,
                ),
                },
                {
                "key": "09-zhang-result",
                "label": "Zhang 2019型（再現実装）結果",
                "image": _draw_result(
                    degraded,
                    ground_truth,
                    zhang_predicted,
                    zhang_evaluation,
                ),
                },
            ])
            method_metadata["zhang2019_arc_reproduction"] = {
                "label": METHOD_LABELS["zhang2019_arc_reproduction"],
                "arc_count": len(zhang_stages["arcs"]),
                "candidate_count": len(zhang_candidates),
                "selection_mode": (
                    zhang_selected.get("selection_mode")
                    if zhang_selected is not None
                    else None
                ),
                "detected": _ellipse_to_dict(zhang_predicted),
                "evaluation": zhang_evaluation,
            }

        if "cnn_ransac" in methods:
            # CNNリング尤度とweighted RANSAC。
            input_size = int(self.cnn_config["input_size"])
            cnn_image = cv2.resize(
                degraded,
                (input_size, input_size),
                interpolation=cv2.INTER_AREA,
            )
            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    cv2.cvtColor(cnn_image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                )
            ).float()
            tensor = (tensor / 127.5 - 1.0)[None].to(self.device)
            with torch.inference_mode():
                probability = (
                    torch.sigmoid(self._load_model()(tensor))[0, 0].cpu().numpy()
                )
            cnn_result = probability_ellipse(
                probability,
                self.cnn_config["cnn_ransac"],
                random_seed=seed,
            )
            cnn_predicted = cnn_result["ellipse"] if cnn_result else None
            scale = input_size / float(label["image_width"])
            cnn_ground_truth = scale_ellipse(ground_truth, scale, scale)
            cnn_evaluation = (
                evaluate_ellipses(
                    cnn_predicted,
                    cnn_ground_truth,
                    probability.shape,
                )
                if cnn_predicted is not None
                else None
            )
            probability_threshold = float(
                self.cnn_config["cnn_ransac"]["probability_threshold"]
            )
            stages.extend([
                {
                "key": "10-cnn-input",
                "label": "CNN入力 256×256",
                "image": cnn_image,
                },
                {
                "key": "11-cnn-probability",
                "label": "CNNリング尤度",
                "image": _probability_heatmap(probability),
                },
                {
                "key": "12-cnn-threshold-points",
                "label": f"閾値点群 ≥ {probability_threshold:.2f}",
                "image": _threshold_points(
                    probability,
                    probability_threshold,
                ),
                },
                {
                "key": "13-cnn-ransac-inliers",
                "label": "weighted RANSACインライア",
                "image": _ransac_inliers(
                    probability,
                    probability_threshold,
                    cnn_predicted,
                    float(
                        self.cnn_config["cnn_ransac"][
                            "distance_threshold_px"
                        ]
                    ),
                ),
                },
                {
                "key": "14-cnn-result",
                "label": "CNN＋weighted RANSAC結果",
                "image": _draw_result(
                    cnn_image,
                    cnn_ground_truth,
                    cnn_predicted,
                    cnn_evaluation,
                ),
                },
            ])
            method_metadata["cnn_ransac"] = {
                "label": METHOD_LABELS["cnn_ransac"],
                "probability_threshold": probability_threshold,
                "detected": _ellipse_to_dict(cnn_predicted),
                "evaluation": cnn_evaluation,
            }

        destination = _project_path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        stage_files = {}
        for stage in stages:
            stage_path = destination / f"{stage['key']}.png"
            if not imwrite(stage_path, stage["image"]):
                raise OSError(f"画像を書き込めませんでした: {stage_path}")
            stage_files[stage["key"]] = str(stage_path)
        overview_path = destination / "overview.png"
        _save_overview(stages, overview_path)

        metadata = {
            "sample_id": sample_id,
            "dataset_dir": str(dataset),
            "source_image": str(dataset / sample["image"]),
            "effect": effect,
            "effect_label": EFFECT_LABELS[effect],
            "severity": severity,
            "random_seed": int(seed),
            "effect_metadata": effect_metadata,
            "selected_methods": methods,
            "methods": method_metadata,
            "stage_files": stage_files,
            "overview": str(overview_path),
        }
        metadata_path = destination / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                _json_ready(metadata),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "output_dir": destination,
            "overview": overview_path,
            "metadata": metadata_path,
            "method_results": method_metadata,
            "stage_files": stage_files,
        }
