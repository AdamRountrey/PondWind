from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from predictweather.runtime import configure_geospatial_runtime

configure_geospatial_runtime()

from predictweather.boundary import load_cached_hrdps_manifest_for_valid_time
from predictweather.gfs import build_gfs_pds_url
from predictweather.gefs import build_gefs_pds_url, build_gefs_subset_url
from predictweather.geo import BoundingBox
from predictweather.hrrr import build_hrrr_pds_url
from predictweather.nam import build_nam_pds_url
from predictweather.noaa_pds import find_grib_message, parse_grib_index
from predictweather.nomads import nomads_filter_url
from predictweather.site import PreparedSiteDomain


class ForecastDownloadHardeningTests(unittest.TestCase):
    def test_gfs_nomads_url_can_use_360_longitudes(self) -> None:
        url = nomads_filter_url(
            filter_script="filter_gfs_0p25.pl",
            file_name="gfs.t12z.pgrb2.0p25.f006",
            dir_path="/gfs.20260517/12/atmos",
            variable_name="UGRD",
            level_flag="lev_10_m_above_ground",
            lat=42.31,
            lon=-83.75,
            lon_360=True,
        )

        self.assertIn("leftlon=275.8000", url)
        self.assertIn("rightlon=276.7000", url)

    def test_hrdps_cache_rejects_wrong_valid_time_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "20260517T06Z" / "PT012H"
            data_dir.mkdir(parents=True)
            u_path = data_dir / "u.grib2"
            v_path = data_dir / "v.grib2"
            u_path.write_bytes(b"u")
            v_path.write_bytes(b"v")
            manifest = {
                "selected_valid_at_utc": "2026-05-17T18:00:00+00:00",
                "run_at_utc": "2026-05-17T06:00:00+00:00",
                "forecast_hour": 12,
                "files": {"UGRD": str(u_path), "VGRD": str(v_path)},
            }
            (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_cached_hrdps_manifest_for_valid_time(
                    root,
                    datetime(2026, 5, 17, 19, tzinfo=timezone.utc),
                )

    def test_noaa_grib_index_selects_message_range(self) -> None:
        entries = parse_grib_index(
            "\n".join(
                [
                    "1:0:d=2026051712:TMP:2 m above ground:6 hour fcst:",
                    "2:128:d=2026051712:UGRD:10 m above ground:6 hour fcst:",
                    "3:512:d=2026051712:VGRD:10 m above ground:6 hour fcst:",
                ]
            )
        )

        entry, next_offset = find_grib_message(entries, variable="UGRD", level_text="10 m above ground")

        self.assertEqual(entry.offset, 128)
        self.assertEqual(next_offset, 512)

    def test_noaa_public_bucket_urls(self) -> None:
        run_at = datetime(2026, 5, 17, 12, tzinfo=timezone.utc)

        self.assertEqual(
            build_gfs_pds_url(run_at, 6),
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260517/12/atmos/gfs.t12z.pgrb2.0p25.f006",
        )
        self.assertEqual(
            build_hrrr_pds_url(run_at, 6),
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260517/conus/hrrr.t12z.wrfsfcf06.grib2",
        )
        self.assertEqual(
            build_nam_pds_url(run_at, 6),
            "https://noaa-nam-pds.s3.amazonaws.com/nam.20260517/nam.t12z.conusnest.hiresf06.tm00.grib2",
        )
        self.assertEqual(
            build_gefs_pds_url("gep03", run_at, 6),
            "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260517/12/atmos/pgrb2ap5/gep03.t12z.pgrb2a.0p50.f006",
        )

    def test_gefs_member_subset_url_requests_both_10m_wind_components(self) -> None:
        run_at = datetime(2026, 5, 17, 12, tzinfo=timezone.utc)

        url = build_gefs_subset_url("gep03", run_at, 6, lat=42.31, lon=-83.75)

        self.assertIn("filter_gefs_atmos_0p50a.pl", url)
        self.assertIn("file=gep03.t12z.pgrb2a.0p50.f006", url)
        self.assertIn("lev_10_m_above_ground=on", url)
        self.assertIn("var_UGRD=on", url)
        self.assertIn("var_VGRD=on", url)
        self.assertIn("leftlon=275.8000", url)
        self.assertIn("rightlon=276.7000", url)

    def test_gefs_indexed_download_falls_back_to_nomads(self) -> None:
        from predictweather import gefs

        run_at = datetime(2026, 5, 17, 12, tzinfo=timezone.utc)
        calls: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "gep03.t12z.pgrb2a.0p50.f006"

            def fail_indexed(path: Path, grib_url: str) -> str:
                calls.append(("pds", grib_url))
                raise RuntimeError("pds unavailable")

            def fake_nomads(url: str, path: Path, **kwargs) -> Path:
                calls.append(("nomads", url))
                path.write_bytes(b"grib")
                return path

            with patch.object(gefs, "_download_gefs_indexed_uv_product", side_effect=fail_indexed), patch.object(
                gefs,
                "download_url_to_file",
                side_effect=fake_nomads,
            ):
                mode = gefs._download_gefs_product(destination, "gep03", run_at, 6)

            self.assertEqual(mode, "nomads_full_file")
            self.assertTrue(destination.exists())
            self.assertEqual(calls[0][0], "pds")
            self.assertEqual(calls[1][0], "nomads")

    def test_representative_gefs_member_selection_is_weighted_cluster_medoid(self) -> None:
        from scripts.run_barton_weekly_report import _select_representative_gefs_members

        members = [
            {"member_id": "gec00", "u10_mps": 0.0, "v10_mps": 5.0, "speed_mps": 5.0},
            {"member_id": "gep01", "u10_mps": 0.2, "v10_mps": 5.1, "speed_mps": 5.1},
            {"member_id": "gep02", "u10_mps": -0.1, "v10_mps": 4.9, "speed_mps": 4.9},
            {"member_id": "gep03", "u10_mps": 5.0, "v10_mps": 0.0, "speed_mps": 5.0},
            {"member_id": "gep04", "u10_mps": 5.2, "v10_mps": -0.1, "speed_mps": 5.2},
            {"member_id": "gep05", "u10_mps": float("nan"), "v10_mps": 2.0, "speed_mps": 2.0},
        ]

        selected = _select_representative_gefs_members(members, max_members=2)

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(member["member_id"].startswith("ge") for member in selected))
        self.assertTrue(all(member["selection_reason"] == "weighted_uv_cluster_medoid" for member in selected))
        self.assertEqual(sum(member["cluster_member_count"] for member in selected), 5)
        self.assertAlmostEqual(sum(member["member_weight"] for member in selected), 5.0)

    def test_weighted_gefs_spread_helpers(self) -> None:
        from scripts.run_barton_weekly_report import _weighted_circular_std_deg, _weighted_nanstd

        stack = np.array([[[0.0]], [[10.0]]], dtype=np.float32)
        weights = np.array([3.0, 1.0], dtype=np.float64)

        self.assertAlmostEqual(float(_weighted_nanstd(stack, weights)[0, 0]), 4.330127, places=5)

        directions = np.array([[[350.0]], [[10.0]]], dtype=np.float32)
        self.assertLess(float(_weighted_circular_std_deg(directions, np.array([1.0, 1.0]))[0, 0]), 15.0)

    def test_parallel_worker_settings_are_bounded_and_configurable(self) -> None:
        from scripts.run_barton_weekly_report import _model_download_workers, _windninja_member_workers

        with patch.dict(
            "os.environ",
            {
                "PONDWIND_MODEL_DOWNLOAD_WORKERS": "6",
                "PONDWIND_WINDNINJA_MEMBER_WORKERS": "3",
            },
            clear=False,
        ):
            self.assertEqual(_model_download_workers(), 6)
            self.assertEqual(_windninja_member_workers(), 3)

        with patch.dict(
            "os.environ",
            {
                "PONDWIND_MODEL_DOWNLOAD_WORKERS": "not-a-number",
                "PONDWIND_WINDNINJA_MEMBER_WORKERS": "0",
            },
            clear=False,
        ):
            self.assertEqual(_model_download_workers(), 4)
            self.assertEqual(_windninja_member_workers(), 1)

    def test_satellite_search_failure_degrades_to_unavailable_inputs(self) -> None:
        from scripts import run_barton_weekly_report as report
        from predictweather.config import SiteConfig

        site = SiteConfig(center_lat=42.31, center_lon=-83.75, side_meters=1000.0)
        domain = PreparedSiteDomain(
            site=site,
            bbox=BoundingBox(-83.76, 42.30, -83.74, 42.32),
            solve_bbox=BoundingBox(-83.77, 42.29, -83.73, 42.33),
            dataset="test",
            source_dem_tif=Path("source.tif"),
            clipped_dem_tif=Path("clip.tif"),
            dem_preview_tif=Path("preview.tif"),
            solve_dem_tif=Path("solve.tif"),
            solve_dem_preview_tif=Path("solve_preview.tif"),
            raw_download_path=Path("raw.zip"),
            manifest_path=Path("manifest.json"),
        )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            report,
            "list_candidate_items",
            side_effect=TimeoutError("provider 504"),
        ):
            result = report._download_satellite_inputs(
                Path(tmp),
                datetime(2026, 5, 17, 18, tzinfo=timezone.utc),
                domain,
                satellite_products={"rgb": True, "sst": True, "chla": True, "turbidity": True},
            )

        self.assertIsNone(result["sentinel_rgb"])
        self.assertIsNone(result["sentinel"])
        self.assertIsNone(result["landsat"])
        self.assertIn("sentinel_rgb_search", result["selection_diagnostics"]["search_errors"])
        self.assertIn("landsat_search", result["selection_diagnostics"]["search_errors"])


if __name__ == "__main__":
    unittest.main()
