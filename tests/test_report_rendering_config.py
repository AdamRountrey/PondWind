from __future__ import annotations

import unittest

from scripts.run_barton_weekly_report import _estimate_openfoam_solve_cells_from_extent, _vector_overlay_style


class ReportRenderingConfigTests(unittest.TestCase):
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
