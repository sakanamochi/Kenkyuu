"""パラメトリックPAF生成器の最小回帰テスト。"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from generate_parametric_paf import generate  # noqa: E402


class ParametricPAFTest(unittest.TestCase):
    def test_target_ring_is_unique_and_matches_metadata(self) -> None:
        config_path = STUDY_ROOT / "config/model_comparison_a.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dimensions = config["dimensions"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.obj"
            metadata = generate(config_path, output)
            vertices = [
                tuple(float(value) for value in line.split()[1:4])
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.startswith("v ")
            ]
            target = [
                vertex
                for vertex in vertices
                if abs(vertex[2] - dimensions["front_z"]) <= 1e-4
                and abs(math.hypot(vertex[0], vertex[1]) - dimensions["inner_radius"])
                <= 1e-4
            ]
            self.assertEqual(len(target), config["segments"])
            self.assertEqual(
                metadata["target_ring"]["vertex_count"],
                config["segments"],
            )
            self.assertTrue(output.with_suffix(".mtl").exists())

    def test_generated_mesh_has_all_material_groups(self) -> None:
        config_path = STUDY_ROOT / "config/model_comparison_a.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.obj"
            generate(config_path, output)
            text = output.read_text(encoding="utf-8")
            for material in (
                "AluminumBody",
                "AluminumRib",
                "PEEKLatch",
                "SteelPost",
            ):
                self.assertIn(f"usemtl {material}", text)


if __name__ == "__main__":
    unittest.main()
