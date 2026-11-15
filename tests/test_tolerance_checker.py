import unittest

from app.core.tolerance_checker import SymmetricTolerance, ToleranceStatus, check_tolerance


class ToleranceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = SymmetricTolerance(nominal_mm=20.0, plus_minus_mm=0.05)

    def test_passes_nominal_and_both_boundaries(self) -> None:
        for measured in (19.95, 20.0, 20.05):
            result = check_tolerance(measured, self.specification)
            self.assertEqual(result.status, ToleranceStatus.PASS)
            self.assertEqual(result.lower_limit_mm, 19.95)
            self.assertEqual(result.upper_limit_mm, 20.05)

    def test_fails_outside_boundaries(self) -> None:
        self.assertEqual(check_tolerance(19.949, self.specification).status, ToleranceStatus.FAIL)
        self.assertEqual(check_tolerance(20.051, self.specification).status, ToleranceStatus.FAIL)

    def test_structured_result(self) -> None:
        self.assertEqual(
            check_tolerance(20.02, self.specification).as_dict(),
            {
                "nominal_mm": 20.0,
                "measured_mm": 20.02,
                "lower_limit_mm": 19.95,
                "upper_limit_mm": 20.05,
                "status": "PASS",
            },
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            SymmetricTolerance(20, -0.05)
        with self.assertRaises(ValueError):
            SymmetricTolerance(0, 0.05)
        with self.assertRaises(ValueError):
            check_tolerance(-0.01, self.specification)


if __name__ == "__main__":
    unittest.main()
