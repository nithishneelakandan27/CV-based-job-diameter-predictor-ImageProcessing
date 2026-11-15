import math
import unittest

from app.core.calibration import LinearScaleCalibration
from app.core.measurement import CircleDiameterMeasurement, PixelDiameterMeasurement


class PixelMeasurementTests(unittest.TestCase):
    def test_pixel_diameter_is_right_minus_left(self) -> None:
        measurement = PixelDiameterMeasurement(left_edge_x=100, right_edge_x=480)
        self.assertEqual(measurement.diameter_pixels, 380)

    def test_fractional_subpixel_edges_are_preserved(self) -> None:
        measurement = PixelDiameterMeasurement(100.25, 480.75)
        self.assertEqual(measurement.diameter_pixels, 380.5)

    def test_rejects_reversed_or_equal_edges(self) -> None:
        for left, right in ((10, 10), (20, 10)):
            with self.assertRaises(ValueError):
                PixelDiameterMeasurement(left, right)

    def test_rejects_non_finite_edge(self) -> None:
        with self.assertRaises(ValueError):
            PixelDiameterMeasurement(math.nan, 10)

    def test_circle_diameter(self) -> None:
        self.assertEqual(CircleDiameterMeasurement(1, 2, 12.5).diameter_pixels, 25.0)

    def test_calibrated_measurement_requires_explicit_model(self) -> None:
        calibrated = PixelDiameterMeasurement(0, 100).calibrated_mm(LinearScaleCalibration(0.02))
        self.assertEqual(calibrated.diameter_mm, 2.0)


class CalibrationTests(unittest.TestCase):
    def test_linear_scale_conversion(self) -> None:
        calibration = LinearScaleCalibration(mm_per_pixel=0.05, reference_description="traceable gauge")
        self.assertEqual(calibration.pixels_to_mm(400), 20.0)

    def test_rejects_invalid_scale_or_distance(self) -> None:
        for scale in (0, -0.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                LinearScaleCalibration(scale)
        calibration = LinearScaleCalibration(0.1)
        for pixels in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                calibration.pixels_to_mm(pixels)


if __name__ == "__main__":
    unittest.main()
