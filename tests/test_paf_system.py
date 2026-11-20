"""PAF内周検出システムの単体・結合・層分離テスト。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defaults import DEFAULT_CONFIG_PATH, DEFAULT_RENDER_DIVERSIFIED_CONFIG
from cg.geometry_model import build_dimensions, generate_diversified_plan, make_geometry
from core.effects import apply_effect
from core.geometry import evaluate_ellipses
from core.io import load_config, read_json, resolve_path
from core.model import TinyUNet, count_parameters, load_checkpoint
from core.ransac import boundary_ellipse, fit_ellipse_ransac, skeleton_points
from inference.canny import detect_canny_contour_ransac
from inference.predictor import PAFDetector
from inference.zhang2019 import ZHANG_2019_REFERENCE, detect_zhang2019
from training.dataset import MultiViewBoundaryDataset
from training.trainer import boundary_loss, boundary_target, resolve_preset_views, train_model


def _synthetic_ring_image(size: int = 256) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / 256.0
    center = (128.0 * scale, 128.0 * scale)
    cv2.ellipse(image, (center, (140.0 * scale, 120.0 * scale), 15.0), (190, 190, 190), -1, cv2.LINE_AA)
    cv2.ellipse(image, (center, (84.0 * scale, 72.0 * scale), 15.0), (20, 20, 20), -1, cv2.LINE_AA)
    return image


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_layer_separation_and_no_legacy_imports() -> None:
    """基本システム(core)・推論(inference)・学習(training)・CG(cg)の依存境界を確認する。"""
    src_root = PROJECT_ROOT / "src"
    for py_file in src_root.rglob("*.py"):
        mods = _imported_modules(py_file)
        assert not any(m.startswith(("paf_ring_detection", "research")) for m in mods), py_file
    for py_file in (src_root / "core").glob("*.py"):
        mods = _imported_modules(py_file)
        assert not any(
            m.startswith(("training", "inference", "evaluation", "cg")) for m in mods
        ), py_file
    for py_file in (src_root / "inference").glob("*.py"):
        mods = _imported_modules(py_file)
        assert not any(m.startswith(("training", "evaluation", "cg")) for m in mods), py_file
    for py_file in (src_root / "training").glob("*.py"):
        mods = _imported_modules(py_file)
        assert not any(m.startswith(("inference", "evaluation", "cg")) for m in mods), py_file


def test_tiny_unet_architecture_and_checkpoint_loading() -> None:
    config, base_dir = load_config(DEFAULT_CONFIG_PATH)
    model = TinyUNet(16)
    assert count_parameters(model) == 482_737
    assert model(torch.zeros((1, 3, 256, 256))).shape == (1, 1, 256, 256)

    models_dir = resolve_path(base_dir, config["paths"]["models_dir"])
    for source in ("1194m", "diversified"):
        ckpt_path = models_dir / source / "20260907" / "best.pt"
        loaded, metadata = load_checkpoint(ckpt_path, torch.device("cpu"))
        assert count_parameters(loaded) == 482_737
        assert metadata["epoch"] > 0


def test_effects_direction_and_bounds() -> None:
    image = _synthetic_ring_image(480)
    mild = apply_effect(image, "sensor_whiteout", 0.5, np.random.default_rng(12345), whiteout_maximum_strength=1.0)
    strong = apply_effect(image, "sensor_whiteout", 1.0, np.random.default_rng(12345), whiteout_maximum_strength=2.0)
    assert strong.mean() >= mild.mean()

    dark = apply_effect(image, "sensor_black_crush", 0.75, np.random.default_rng(12345))
    assert dark.mean() < image.mean()

    occluded = apply_effect(image, "black_rectangle", 0.5, np.random.default_rng(12345))
    assert (occluded == 0).sum() > (image == 0).sum()


def test_geometry_ransac_and_skeleton() -> None:
    truth = ((128.0, 128.0), (80.0, 60.0), 25.0)
    assert evaluate_ellipses(truth, truth, (256, 256))["ellipse_iou"] == pytest.approx(1.0)
    shifted = ((150.0, 128.0), (80.0, 60.0), 25.0)
    assert evaluate_ellipses(shifted, truth, (256, 256))["ellipse_iou"] < 0.8

    prob = np.zeros((256, 256), dtype=np.float32)
    cv2.ellipse(prob, truth, 1.0, 3)
    pts, weights = skeleton_points(prob, threshold=0.5)
    assert len(pts) >= 40
    assert np.all(weights >= 0.5)

    config, _ = load_config(DEFAULT_CONFIG_PATH)
    fitted = fit_ellipse_ransac(pts, prob.shape, config["ransac_cnn"], weights=weights)
    assert fitted is not None
    assert evaluate_ellipses(fitted["ellipse"], truth, (256, 256))["ellipse_iou"] > 0.9

    detected = boundary_ellipse(prob, config["ransac_cnn"])
    assert detected is not None
    assert evaluate_ellipses(detected, truth, (256, 256))["ellipse_iou"] > 0.9


def test_geometric_baselines_and_zhang_citation() -> None:
    config, _ = load_config(DEFAULT_CONFIG_PATH)
    image = _synthetic_ring_image(256)
    truth = ((128.0, 128.0), (84.0, 72.0), 15.0)

    canny_sel, _ = detect_canny_contour_ransac(
        image, config["canny_contour_ransac"], config["ransac_geometry"], 2026102404
    )
    assert canny_sel is not None
    assert evaluate_ellipses(canny_sel["ellipse"], truth, (256, 256))["ellipse_iou"] > 0.9

    zhang_sel, _ = detect_zhang2019(image, config["zhang2019"])
    assert zhang_sel is not None
    assert evaluate_ellipses(zhang_sel["ellipse"], truth, (256, 256))["ellipse_iou"] > 0.9
    assert ZHANG_2019_REFERENCE["doi"] == "10.3390/s19235243"
    assert "再現実装" in ZHANG_2019_REFERENCE["label"]


def test_parametric_cg_model_and_plan(tmp_path: Path) -> None:
    cfg = read_json(DEFAULT_RENDER_DIVERSIFIED_CONFIG)
    dims = build_dimensions(
        cfg,
        {"bottom_outer_diameter_ratio": 2.0, "height_ratio": 1.5, "hole_diameter_ratio": 0.7},
    )
    vertices, faces, rim, profile = make_geometry(dims, int(cfg["segments"]))
    assert len(rim) == int(cfg["segments"])
    assert len(vertices) > 0 and len(faces) > 0 and len(profile) == 8

    with pytest.raises(ValueError):
        make_geometry({**dims, "hole_diameter": 0.99}, 64)

    plan = generate_diversified_plan(DEFAULT_RENDER_DIVERSIFIED_CONFIG, tmp_path / "cg_plan")
    assert len(plan["samples"]) == int(cfg["image_count"])
    assert (tmp_path / "cg_plan" / "plan.json").exists()


def test_multiview_dataset_and_boundary_target() -> None:
    config, base_dir = load_config(DEFAULT_CONFIG_PATH)
    for source_name in ("1194m", "diversified"):
        preset, view_dirs = resolve_preset_views(config, base_dir, source_name)
        ds = MultiViewBoundaryDataset(
            view_dirs=view_dirs,
            view_slots=preset["view_slots"],
            split="train",
            input_size=256,
            cache=True,
            seed=20260907,
            limit=32,
        )
        assert len(ds) == 32
        img, masks, idx = ds[0]
        assert img.shape == (3, 256, 256)
        assert masks.shape == (2, 256, 256)
        assert idx == 0

        target = boundary_target(masks[None])
        assert target.shape == (1, 1, 256, 256)
        assert float(target.min()) >= 0.0 and float(target.max()) <= 1.0
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]))
        loss = boundary_loss(torch.zeros((1, 1, 256, 256)), masks[None], bce)
        assert torch.isfinite(loss)


def test_unified_predictor_uniform_rejection_and_padding(tmp_path: Path) -> None:
    config, base_dir = load_config(DEFAULT_CONFIG_PATH)
    detector = PAFDetector(config, base_dir, method="cnn_diversified", seed=20260907, device_name="cpu")

    blank_res = detector.predict_array(np.zeros((256, 256, 3), dtype=np.uint8))
    assert blank_res["status"] == "insufficient_image_evidence"
    assert blank_res["ellipse"] is None

    rect = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.predict_array(rect, pad_non_square=False)
    rect_res = detector.predict_array(rect, pad_non_square=True)
    assert rect_res["padded_side"] == 160
    assert rect_res["status"] == "insufficient_image_evidence"

    first_png = sorted(resolve_path(base_dir, "data/1194m/base/images").glob("*.png"))[0]
    saved = detector.predict_file(first_png, tmp_path / "pred")
    assert saved["status"] == "detected"
    assert (tmp_path / "pred" / "result.json").exists()
    assert (tmp_path / "pred" / "overlay.png").exists()
    assert (tmp_path / "pred" / "probability.npy").exists()


def test_training_smoke_one_epoch(tmp_path: Path) -> None:
    config, base_dir = load_config(DEFAULT_CONFIG_PATH)
    summary = train_model(
        config,
        base_dir,
        source_name="diversified",
        seed=20260907,
        output_dir=tmp_path / "smoke_run",
        epochs=1,
        device_name="cpu",
        limit_samples=8,
    )
    assert summary["epochs"] == 1
    assert summary["best_epoch"] == 1
    assert (tmp_path / "smoke_run" / "best.pt").exists()
    assert (tmp_path / "smoke_run" / "complete.json").exists()
    assert (tmp_path / "smoke_run" / "training_history.json").exists()
