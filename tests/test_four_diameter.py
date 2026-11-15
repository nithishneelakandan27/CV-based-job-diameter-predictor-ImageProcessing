import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.core.calibration import LinearScaleCalibration
from app.core.diameter_engine import DiameterStatus, FourDiameterMeasurementEngine
from app.core.inspector import DiameterInspectionEngine
from app.core.job_detector import FixedRoiJobDetector, JobROI
from app.core.measurement_spec import DiameterSpec, FourDiameterSpec, MeasurementRegion
from scripts.run_four_diameter_experiment import run as run_four_diameter_experiment


def synthetic_stepped_job() -> np.ndarray:
    """Four white stepped sections on black, aligned for deterministic tests."""

    image = np.zeros((200, 120, 3), dtype=np.uint8)
    # D1 largest, D2 smaller, D3/D4 equal. Bands avoid the horizontal steps.
    for start_y, end_y, left, right in ((0, 50, 10, 110), (50, 100, 20, 100), (100, 150, 25, 95), (150, 200, 25, 95)):
        cv2.rectangle(image, (left, start_y), (right, end_y - 1), (255, 255, 255), -1)
    return image


def spec(with_tolerance: bool = False) -> FourDiameterSpec:
    tolerance_values = {"D1": (20.0, 0.5), "D2": (16.0, 0.5), "D3": (14.0, 0.5), "D4": (14.0, 0.5)}
    bands = {"D1": (0.05, 0.20), "D2": (0.30, 0.45), "D3": (0.55, 0.70), "D4": (0.80, 0.95)}
    return FourDiameterSpec(tuple(
        DiameterSpec(
            name=name,
            region=MeasurementRegion(*bands[name]),
            nominal_mm=tolerance_values[name][0] if with_tolerance else None,
            tolerance_mm=tolerance_values[name][1] if with_tolerance else None,
        )
        for name in ("D1", "D2", "D3", "D4")
    ))


class FourDiameterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FourDiameterMeasurementEngine()
        self.image = synthetic_stepped_job()

    def test_d1_d2_d3_d4_are_measured_in_independent_regions(self) -> None:
        results = self.engine.measure_job(self.image, spec()).diameters
        by_name = {result.name: result for result in results}
        self.assertEqual(set(by_name), {"D1", "D2", "D3", "D4"})
        self.assertGreater(by_name["D1"].diameter_pixels, by_name["D2"].diameter_pixels)
        self.assertGreater(by_name["D2"].diameter_pixels, by_name["D3"].diameter_pixels)
        self.assertAlmostEqual(by_name["D3"].diameter_pixels, by_name["D4"].diameter_pixels, delta=1.0)
        self.assertIsNot(by_name["D3"], by_name["D4"])

    def test_missing_calibration_and_specification_is_not_configured(self) -> None:
        for result in self.engine.measure_job(self.image, spec()).diameters:
            self.assertEqual(result.status, DiameterStatus.NOT_CONFIGURED)
            self.assertIsNotNone(result.diameter_pixels)
            self.assertIsNone(result.diameter_mm)

    def test_missing_calibration_with_specification_is_not_configured(self) -> None:
        result = self.engine.measure_job(self.image, spec(with_tolerance=True)).diameters[0]
        self.assertEqual(result.status, DiameterStatus.NOT_CONFIGURED)
        self.assertIsNone(result.diameter_mm)
        self.assertEqual(result.lower_limit_mm, 19.5)
        self.assertEqual(result.upper_limit_mm, 20.5)

    def test_pixel_to_mm_and_equal_d3_d4_nominals_remain_independent(self) -> None:
        measurement = self.engine.measure_job(self.image, spec(with_tolerance=True), LinearScaleCalibration(0.2))
        by_name = {result.name: result for result in measurement.diameters}
        self.assertEqual(by_name["D3"].nominal_mm, by_name["D4"].nominal_mm)
        self.assertIsNotNone(by_name["D3"].diameter_mm)
        self.assertIsNotNone(by_name["D4"].diameter_mm)
        self.assertNotEqual(by_name["D3"].region, by_name["D4"].region)

    def test_calibration_without_tolerance_keeps_mm_and_not_configured_status(self) -> None:
        measurement = self.engine.measure_job(self.image, spec(), LinearScaleCalibration(0.2))
        self.assertEqual(measurement.status, DiameterStatus.NOT_CONFIGURED)
        for result in measurement.diameters:
            self.assertIsNotNone(result.diameter_mm)
            self.assertEqual(result.status, DiameterStatus.NOT_CONFIGURED)

    def test_job_status_aggregates_pass_fail_not_configured_and_measurement_failure(self) -> None:
        self.assertEqual(
            self.engine.measure_job(self.image, spec(with_tolerance=True), LinearScaleCalibration(0.2)).status,
            DiameterStatus.PASS,
        )
        self.assertEqual(
            self.engine.measure_job(self.image, spec(with_tolerance=True), LinearScaleCalibration(0.1)).status,
            DiameterStatus.FAIL,
        )
        self.assertEqual(
            self.engine.measure_job(self.image, spec(with_tolerance=True)).status,
            DiameterStatus.NOT_CONFIGURED,
        )
        bad_d1 = DiameterSpec("D1", MeasurementRegion(0.05, 0.20), edge_strategy="not-a-method")
        valid = spec().ordered()
        failed_spec = FourDiameterSpec((bad_d1, valid[1], valid[2], valid[3]))
        self.assertEqual(
            self.engine.measure_job(self.image, failed_spec, LinearScaleCalibration(0.2)).status,
            DiameterStatus.MEASUREMENT_FAILED,
        )

    def test_invalid_regions_and_edges_are_rejected_or_reported(self) -> None:
        with self.assertRaises(ValueError):
            MeasurementRegion(0.4, 0.4)
        with self.assertRaises(ValueError):
            MeasurementRegion(-0.1, 0.2)
        bad_spec = DiameterSpec("D1", MeasurementRegion(0.0, 0.2), edge_strategy="not-a-method")
        result = self.engine.measure_diameter(self.image, bad_spec)
        self.assertEqual(result.status, DiameterStatus.MEASUREMENT_FAILED)

    def test_fixed_roi_detector_rejects_out_of_bounds_roi_on_extraction(self) -> None:
        detector = FixedRoiJobDetector((JobROI("job-1", 100, 100, 30, 30),))
        with self.assertRaises(ValueError):
            detector.locate(self.image)[0].extract(self.image)

    def test_complete_configured_image_times_every_required_stage(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "synthetic.png"
            self.assertTrue(cv2.imwrite(str(image_path), self.image))
            inspection = DiameterInspectionEngine().inspect_configured_image(
                image_path,
                FixedRoiJobDetector((JobROI("job-1", 0, 0, 120, 200),)),
                spec(),
            )
        self.assertEqual(len(inspection.jobs), 1)
        self.assertEqual([item.name for item in inspection.jobs[0].result.diameters], ["D1", "D2", "D3", "D4"])
        for stage in (
            "job_localization", "roi_extraction", "normalization", "D1_measurement",
            "D2_measurement", "D3_measurement", "D4_measurement", "complete_image_inspection",
        ):
            self.assertIn(stage, inspection.timings_ms)

    def test_manual_config_experiment_writes_pixel_only_annotation_and_report(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "synthetic.png"
            config_path = temporary_path / "manual_config.json"
            output_path = temporary_path / "output"
            self.assertTrue(cv2.imwrite(str(image_path), self.image))
            config_path.write_text(json.dumps({
                "job_rois": [{"job_id": "job-1", "x": 0, "y": 0, "width": 120, "height": 200}],
                "diameters": [
                    {"name": "D1", "region": {"top_fraction": 0.05, "bottom_fraction": 0.20}},
                    {"name": "D2", "region": {"top_fraction": 0.30, "bottom_fraction": 0.45}},
                    {"name": "D3", "region": {"top_fraction": 0.55, "bottom_fraction": 0.70}},
                    {"name": "D4", "region": {"top_fraction": 0.80, "bottom_fraction": 0.95}},
                ],
            }), encoding="utf-8")
            run_four_diameter_experiment(image_path, config_path, output_path)
            report = json.loads((output_path / "four_diameter_experiment_report.json").read_text(encoding="utf-8"))
            self.assertTrue((output_path / "synthetic_four_diameter_experiment.png").is_file())
        self.assertEqual([item["status"] for item in report["jobs"][0]["diameters"]], ["NOT_CONFIGURED"] * 4)
        self.assertTrue(all(item["diameter_mm"] is None for item in report["jobs"][0]["diameters"]))


if __name__ == "__main__":
    unittest.main()
