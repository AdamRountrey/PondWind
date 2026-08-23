from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_barton_weekly_report as report
from scripts.run_barton_weekly_report import _estimate_openfoam_solve_cells_from_extent, _vector_overlay_style


class ReportRenderingConfigTests(unittest.TestCase):
    def test_packaged_default_report_root_is_visible_documents_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_root = Path(tmp)
            documents = user_root / "Documents"
            documents.mkdir()
            with patch.object(report.sys, "frozen", True, create=True), patch.object(
                report.Path,
                "home",
                return_value=user_root,
            ):
                resolved = report._resolve_report_root(None)

            self.assertEqual(resolved, documents / "PondWind Reports")

    def test_report_downloads_are_kept_beside_selected_report_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            report,
            "prepare_site_domain",
            side_effect=RuntimeError("stop after storage setup"),
        ) as prepare_domain:
            selected_root = Path(tmp) / "Selected Reports"

            with self.assertRaisesRegex(RuntimeError, "stop after storage setup"):
                report.build_weekly_report(
                    race_local_datetime="2026-08-23T14:00:00",
                    report_output_dir=selected_root,
                    satellite_rgb=False,
                    satellite_sst=False,
                    satellite_chla=False,
                    satellite_turbidity=False,
                )

            call = prepare_domain.call_args
            self.assertEqual(call.kwargs["raw_data_dir"], selected_root / "PondWind Working Data" / "downloads")
            self.assertEqual(call.kwargs["processed_data_dir"], selected_root / "PondWind Working Data" / "processed")
            self.assertTrue((selected_root / "20260823_1400_barton_pond" / "report_temp").is_dir())

    def test_vector_overlay_keeps_30m_visual_density_for_finer_solves(self) -> None:
        self.assertEqual(_vector_overlay_style(30.0), (4, 2.2))
        self.assertEqual(_vector_overlay_style(15.0), (8, 4.4))
        self.assertEqual(_vector_overlay_style(10.0), (12, 6.6000000000000005))
        self.assertEqual(_vector_overlay_style(60.0), (4, 2.2))

    def test_openfoam_solve_cell_estimate_uses_vertical_cells_and_mesh_cap(self) -> None:
        estimate = _estimate_openfoam_solve_cells_from_extent(
            width_m=2400.0,
            height_m=1800.0,
            mesh_resolution_m=10.0,
            vertical_cells=20,
            max_horizontal_cells=12000,
        )

        self.assertLessEqual(estimate["horizontal_cells"], 12000)
        self.assertEqual(estimate["solve_cells"], estimate["horizontal_cells"] * 20)
        self.assertGreater(estimate["nx"], 2)
        self.assertGreater(estimate["ny"], 2)


if __name__ == "__main__":
    unittest.main()
