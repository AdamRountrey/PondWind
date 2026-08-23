from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.predictweather_gui import PredictWeatherGui


class GuiReportLocationTests(unittest.TestCase):
    def test_report_location_chooser_uses_folder_dialog_and_remembers_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            initial_dir = Path(tmp)
            selected_dir = initial_dir / "PondWind Reports"
            gui = object.__new__(PredictWeatherGui)
            gui.root = object()
            gui.report_output_dir_var = Mock()
            gui.report_output_dir_var.get.return_value = str(initial_dir)

            with patch(
                "scripts.predictweather_gui.filedialog.askdirectory",
                return_value=str(selected_dir),
            ) as askdirectory:
                selected = gui._choose_report_output_dir()

            self.assertEqual(selected, str(selected_dir))
            gui.report_output_dir_var.set.assert_called_once_with(str(selected_dir))
            self.assertEqual(askdirectory.call_args.kwargs["initialdir"], str(initial_dir))
            self.assertEqual(askdirectory.call_args.kwargs["title"], "Choose where PondWind should save the report")


if __name__ == "__main__":
    unittest.main()
