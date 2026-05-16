from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from predictweather.openfoam import (
    OpenFoamRunError,
    _default_max_horizontal_cells,
    _openfoam_wsl_preflight,
    compare_wind_outputs,
    run_openfoam_domain_average,
    validate_wind_outputs,
)
from scripts import openfoam_wsl_terrain_runner as runner


def _write_aaigrid(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.where(np.isfinite(data), data, -9999.0)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ncols 2\n")
        handle.write("nrows 2\n")
        handle.write("xllcorner 0\n")
        handle.write("yllcorner 0\n")
        handle.write("cellsize 30\n")
        handle.write("NODATA_value -9999\n")
        for row in rows:
            handle.write(" ".join(f"{value:.6f}" for value in row) + "\n")


class OpenFoamContractTests(unittest.TestCase):
    def test_validate_wind_outputs_rejects_diverged_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speed = root / "speed.asc"
            direction = root / "direction.asc"
            _write_aaigrid(speed, np.array([[4.0, 5.0], [7.0, 754778039.0]], dtype=np.float32))
            _write_aaigrid(direction, np.array([[215.0, 215.0], [215.0, 215.0]], dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "sanity limit"):
                validate_wind_outputs(
                    paths={"speed": speed, "direction": direction},
                    boundary_wind_speed_mps=7.0,
                    max_output_speed_mps=75.0,
                )

    def test_compare_wind_outputs_calculates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            windninja = {
                "speed": root / "wn_speed.asc",
                "direction": root / "wn_dir.asc",
            }
            openfoam = {
                "speed": root / "of_speed.asc",
                "direction": root / "of_dir.asc",
            }
            _write_aaigrid(windninja["speed"], np.full((2, 2), 5.0, dtype=np.float32))
            _write_aaigrid(windninja["direction"], np.full((2, 2), 270.0, dtype=np.float32))
            _write_aaigrid(openfoam["speed"], np.full((2, 2), 6.0, dtype=np.float32))
            _write_aaigrid(openfoam["direction"], np.full((2, 2), 270.0, dtype=np.float32))

            metrics = compare_wind_outputs(
                windninja_paths=windninja,
                openfoam_paths=openfoam,
                water_mask=np.array([[True, False], [True, False]]),
            )

            self.assertEqual(metrics["full_domain"]["finite_pair_count"], 4)
            self.assertAlmostEqual(metrics["full_domain"]["speed_bias_mps"], 1.0)
            self.assertAlmostEqual(metrics["full_domain"]["speed_mae_mps"], 1.0)
            self.assertEqual(metrics["water_only"]["finite_pair_count"], 2)
            self.assertAlmostEqual(metrics["full_domain"]["speed_ratio_openfoam_to_windninja"]["median"], 1.2)

    def test_case_generation_uses_abl_boundaries_and_rough_wall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            terrain_info = {
                "nx": 1,
                "ny": 1,
                "dx": 30.0,
                "dy": 30.0,
                "left": 0.0,
                "bottom": 0.0,
                "cellsize": 30.0,
                "terrain": np.zeros((2, 2), dtype=np.float32),
                "roughness": {"z0_m": 0.03},
            }

            runner._write_case_files(case_dir, terrain_info, wind_u=-5.0, wind_v=0.0, reference_height_m=10.0)

            u_text = (case_dir / "0" / "U").read_text(encoding="utf-8")
            k_text = (case_dir / "0" / "k").read_text(encoding="utf-8")
            epsilon_text = (case_dir / "0" / "epsilon").read_text(encoding="utf-8")
            nut_text = (case_dir / "0" / "nut").read_text(encoding="utf-8")
            abl_dict = (case_dir / "system" / "setAtmBoundaryLayerDict").read_text(encoding="utf-8")

            self.assertIn("atmBoundaryLayerInletVelocity", u_text)
            self.assertIn("atmBoundaryLayerInletK", k_text)
            self.assertIn("atmBoundaryLayerInletEpsilon", epsilon_text)
            self.assertIn("nutkAtmRoughWallFunction", nut_text)
            self.assertIn("z0 uniform 0.03000000", nut_text)
            self.assertIn("Zref 10.00000000", abl_dict)
            sample_text = (case_dir / "system" / "sampleDict").read_text(encoding="utf-8")
            self.assertIn("type sets;", sample_text)
            self.assertIn('libs ("libsampling.so");', sample_text)
            self.assertIn("fields (U k);", sample_text)

    def test_case_generation_writes_nonuniform_terrain_roughness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            terrain_info = {
                "nx": 2,
                "ny": 2,
                "dx": 30.0,
                "dy": 30.0,
                "left": 0.0,
                "bottom": 0.0,
                "cellsize": 30.0,
                "terrain": np.zeros((3, 3), dtype=np.float32),
                "roughness": {"z0_m": 0.03},
                "roughness_z0_grid": np.array([[0.0002, 0.03], [0.12, 0.3]], dtype=np.float32),
            }

            runner._write_case_files(case_dir, terrain_info, wind_u=-5.0, wind_v=0.0, reference_height_m=10.0)

            nut_text = (case_dir / "0" / "nut").read_text(encoding="utf-8")
            self.assertIn("z0 nonuniform List<scalar>", nut_text)
            self.assertIn("4\n            (", nut_text)
            self.assertIn("0.00020000", nut_text)
            self.assertIn("0.30000001", nut_text)

    def test_parse_openfoam13_sets_sample_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "windGrid.xy"
            sample_path.write_text(
                "\n".join(
                    [
                        "# distance x y z U_x U_y U_z",
                        "0 10 20 30 1.5 -2.0 0.1",
                        "30 40 20 32 1.6 -2.1 0.2",
                    ]
                ),
                encoding="utf-8",
            )

            vectors = runner._parse_sample_vectors(sample_path, expected_count=2)

            np.testing.assert_allclose(vectors, np.array([[1.5, -2.0, 0.1], [1.6, -2.1, 0.2]], dtype=np.float32))

    def test_parse_openfoam13_sample_output_with_tke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "windGrid.xy"
            sample_path.write_text(
                "\n".join(
                    [
                        "# distance x y z U_x U_y U_z k",
                        "0 10 20 30 3.0 4.0 0.1 0.375",
                        "30 40 20 32 4.0 3.0 0.2 0.135",
                    ]
                ),
                encoding="utf-8",
            )

            vectors, k_values = runner._parse_sample_fields(sample_path, expected_count=2)

            np.testing.assert_allclose(vectors, np.array([[3.0, 4.0, 0.1], [4.0, 3.0, 0.2]], dtype=np.float32))
            np.testing.assert_allclose(k_values, np.array([0.375, 0.135], dtype=np.float32))

    def test_openfoam_cell_cap_scales_with_requested_grid_size(self) -> None:
        self.assertEqual(_default_max_horizontal_cells(30.0), 12000)
        self.assertEqual(_default_max_horizontal_cells(15.0), 48000)
        self.assertEqual(_default_max_horizontal_cells(10.0), 108000)

    def test_openfoam_missing_runner_is_user_friendly_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"PONDWIND_OPENFOAM_RUNNER": ""}, clear=False):
            with self.assertRaises(OpenFoamRunError) as caught:
                run_openfoam_domain_average(
                    elevation_tif=Path(tmp) / "missing.tif",
                    output_dir=Path(tmp) / "openfoam",
                    wind_speed_mps=5.0,
                    wind_direction_deg=270.0,
                    mesh_resolution_m=30.0,
                )

            self.assertEqual(caught.exception.stage, "openfoam_unavailable")
            self.assertTrue(caught.exception.details["skipped"])
            self.assertIn("without CFD", caught.exception.message)

    def test_custom_openfoam_runner_skips_wsl_preflight(self) -> None:
        availability = _openfoam_wsl_preflight(["custom-openfoam-runner"])

        self.assertTrue(availability["available"])
        self.assertEqual(availability["mode"], "custom_runner")


if __name__ == "__main__":
    unittest.main()
