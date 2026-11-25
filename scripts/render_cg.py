"""CG原画像生成・検査用スクリプト：Blender経由で1194M・任意CADモデル・多様形状の原CGを生成・検証する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cg.cad_dataset import build_cad_evaluation_manifest, inspect_cad_obj, verify_cad_dataset
from cg.geometry_model import build_split_manifests, generate_diversified_plan, verify_diversified_dataset
from core.io import load_config, read_json, resolve_path
from defaults import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RENDER_1194M_CONFIG,
    DEFAULT_RENDER_DIVERSIFIED_CONFIG,
)


def _run_blender(
    blender_path: Path,
    script_path: Path,
    render_config: Path,
    base_dir: Path,
    output_dir: Path | None,
    limit: int | None,
    views: list[str] | None = None,
) -> None:
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender実行ファイルが見つかりません: {blender_path}")
    command = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--config",
        str(render_config),
        "--base-dir",
        str(base_dir),
    ]
    if output_dir is not None:
        command.extend(["--output", str(output_dir.resolve())])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if views:
        command.append("--views")
        command.extend(views)
    subprocess.run(command, check=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="1194M・任意CADモデル・多様形状のCGデータセットを生成・検証します。")
    parser.add_argument("target", choices=["1194m", "cad", "diversified"], help="生成対象（1194m / cad / diversified）")
    parser.add_argument(
        "--stage",
        choices=["inspect", "plan", "render", "manifest", "verify"],
        default="render",
        help="実行段階（cadはinspect / render / manifest / verify、diversifiedはplan / render / verify）",
    )
    parser.add_argument("--model", type=Path, default=None, help="cad inspect用のOBJモデルパス")
    parser.add_argument("--output", type=Path, default=None, help="出力先ディレクトリ（省略時は設定ファイルの既定先）")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に描画枚数を先頭N枚へ制限する")
    parser.add_argument("--views", nargs="+", default=None, help="diversifiedで生成・manifest化するビュー名一覧")
    parser.add_argument("--blender", type=Path, default=None, help="Blender実行ファイルのパス")
    parser.add_argument("--render-config", type=Path, default=None, help="描画設定JSONのパス")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="基本設定JSONのパス")
    args = parser.parse_args()

    exp_config, base_dir = load_config(args.config)
    blender_path = Path(exp_config["blender_executable"]) if args.blender is None else args.blender

    if args.target == "1194m":
        if args.stage != "render":
            raise ValueError("1194mでは --stage render のみ対応しています。")
        render_cfg = (DEFAULT_RENDER_1194M_CONFIG if args.render_config is None else args.render_config).resolve()
        script = PROJECT_ROOT / "src" / "cg" / "render_1194m.py"
        _run_blender(blender_path, script, render_cfg, base_dir, args.output, args.limit)
        return

    if args.target == "cad":
        if args.stage == "inspect":
            if args.model is None:
                raise ValueError("cad --stage inspect には --model <OBJパス> が必要です。")
            info = inspect_cad_obj(resolve_path(base_dir, args.model))
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return
        if args.render_config is None:
            raise ValueError("cadでは --render-config <描画設定JSON> を指定してください。")
        render_cfg = resolve_path(base_dir, args.render_config)
        cfg_data = read_json(render_cfg)
        out_dir = resolve_path(base_dir, cfg_data["output_dir"]) if args.output is None else args.output.resolve()
        if args.stage == "render":
            script = PROJECT_ROOT / "src" / "cg" / "render_1194m.py"
            _run_blender(blender_path, script, render_cfg, base_dir, out_dir, args.limit)
        elif args.stage == "manifest":
            source_name = str(cfg_data.get("source_name", cfg_data["dataset_id"]))
            seed = int(cfg_data.get("evaluation_manifest_seed", exp_config["evaluation_seed"]))
            dest = build_cad_evaluation_manifest(out_dir, source_name=source_name, seed=seed)
            print(f"MANIFEST saved: {dest}")
        elif args.stage == "verify":
            result = verify_cad_dataset(render_cfg, out_dir)
            print(f"VERIFIED: {result}")
        else:
            raise ValueError("cadでは --stage inspect / render / manifest / verify のみ対応しています。")
        return

    render_cfg = (DEFAULT_RENDER_DIVERSIFIED_CONFIG if args.render_config is None else args.render_config).resolve()
    cfg_data = read_json(render_cfg)
    out_dir = resolve_path(base_dir, cfg_data["output_dir"]) if args.output is None else args.output.resolve()

    if args.stage == "plan":
        plan = generate_diversified_plan(render_cfg, out_dir)
        print(f"PLAN saved: {out_dir / 'plan.json'} (samples={len(plan['samples'])})")
    elif args.stage == "manifest":
        views = args.views or list(exp_config["datasets"]["diversified"]["views"].keys())
        written = build_split_manifests(out_dir, views)
        print(f"MANIFESTS saved: {len(written)} views under {out_dir}")
    elif args.stage == "verify":
        result = verify_diversified_dataset(render_cfg, out_dir)
        print(f"VERIFIED: {result}")
    elif args.stage == "render":
        script = PROJECT_ROOT / "src" / "cg" / "render_diversified.py"
        _run_blender(blender_path, script, render_cfg, base_dir, out_dir, args.limit, views=args.views)
    else:
        raise ValueError("diversifiedでは --stage plan / render / manifest / verify のみ対応しています。")


if __name__ == "__main__":
    main()
