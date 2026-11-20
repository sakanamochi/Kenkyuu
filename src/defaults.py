"""プロジェクト基準パスと既定設定ファイルの定義。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
DEFAULT_RENDER_1194M_CONFIG = PROJECT_ROOT / "configs" / "render_1194m.json"
DEFAULT_RENDER_DIVERSIFIED_CONFIG = PROJECT_ROOT / "configs" / "render_diversified.json"
