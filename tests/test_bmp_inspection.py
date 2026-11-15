import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.image_loader import load_bgr_image
from app.core.job_detector import DarkRedSilhouetteJobDetector
from scripts.run_bmp_inspection import inspect_image, load_measurement_configuration, run, write_image_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BMP_IMAGES = sorted((PROJECT_ROOT / "data" / "input").glob("Image_*.bmp"))
EXPECTED_IDS = [
    "job_r01_c01", "job_r01_c02", "job_r01_c03",
    "job_r02_c01", "job_r02_c02", "job_r02_c03",
    "job_r03_c01", "job_r03_c02", "job_r03_c03",
]


class BmpInspectionTests(unittest.TestCase):
    def test_all_fixture_bmps_have_deterministic_three_by_three_jobs_and_measurements(self) -> None:
        self.assertEqual(len(BMP_IMAGES), 4)
        detector = DarkRedSilhouetteJobDetector()
        for image_path in BMP_IMAGES:
            with self.subTest(image=image_path.name):
                self.assertEqual([job.job_id for job in detector.locate(load_bgr_image(image_path))], EXPECTED_IDS)
                inspection = inspect_image(image_path)
                self.assertEqual([job.job_id for job in inspection.jobs], EXPECTED_IDS)
                for job in inspection.jobs:
                    results = job.result.diameters
                    self.assertEqual([result.name for result in results], ["D1", "D2", "D3", "D4"])
                    measurements = [result.diameter_pixels for result in results]
                    self.assertTrue(all(value is not None and value > 0 for value in measurements))
                    self.assertGreater(measurements[0], measurements[1])
                    self.assertGreater(measurements[1], measurements[2])
                    self.assertGreater(measurements[2], measurements[3])

    def test_per_image_annotation_json_and_csv_include_all_jobs(self) -> None:
        image_path = BMP_IMAGES[0]
        with TemporaryDirectory() as temporary_directory:
            paths = write_image_outputs(image_path, Path(temporary_directory), inspect_image(image_path))
            self.assertTrue(all(path.is_file() for path in paths.values()))
            report = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(report["job_count"], 9)
            self.assertEqual(report["status_counts"], {
                "PASS": 0, "FAIL": 0, "NOT_CONFIGURED": 9, "MEASUREMENT_FAILED": 0,
            })
            self.assertEqual(len(report["jobs"]), 9)
            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 36)

    def test_bmp_runner_uses_explicit_synthetic_calibration_and_tolerances(self) -> None:
        image_path = BMP_IMAGES[0]
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            config_path = temporary_path / "synthetic_measurement_config.json"
            output_path = temporary_path / "output"
            config_path.write_text(json.dumps({
                "calibration": {
                    "mm_per_pixel": 0.1,
                    "reference_description": "SYNTHETIC test data only",
                },
                "diameters": [
                    {"name": "D1", "nominal_mm": 38.0, "tolerance_mm": 1.0},
                    {"name": "D2", "nominal_mm": 30.0, "tolerance_mm": 1.0},
                    {"name": "D3", "nominal_mm": 27.0, "tolerance_mm": 1.0},
                    {"name": "D4", "nominal_mm": 22.5, "tolerance_mm": 1.0},
                ],
            }), encoding="utf-8")
            run(image_path, output_path, benchmark_runs=1, measurement_config=config_path)
            report = json.loads((output_path / f"{image_path.stem}_results.json").read_text(encoding="utf-8"))
            with (output_path / f"{image_path.stem}_results.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(report["measurement_units"], "pixels_and_mm")
        self.assertEqual(report["calibration"]["mm_per_pixel"], 0.1)
        self.assertEqual(report["status_counts"], {
            "PASS": 9, "FAIL": 0, "NOT_CONFIGURED": 0, "MEASUREMENT_FAILED": 0,
        })
        self.assertTrue(all(job["status"] == "PASS" for job in report["jobs"]))
        self.assertTrue(all(diameter["diameter_mm"] is not None for job in report["jobs"] for diameter in job["diameters"]))
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(row["job_status"] == "PASS" and row["lower_limit_mm"] for row in rows))

    def test_measurement_config_rejects_invalid_calibration(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid_calibration.json"
            config_path.write_text(json.dumps({"calibration": {"mm_per_pixel": 0}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_measurement_configuration(config_path)


if __name__ == "__main__":
    unittest.main()
