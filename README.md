# Industrial dimensional inspection foundation

This repository is the classical-computer-vision foundation for an industrial
machine-vision system. Its intended end state is to inspect roughly 50 fixtured
components in a GigE-camera image, locate defined measurement regions, measure
dimensions such as diameters, calibrate them to millimetres, and compare them
with engineering tolerances.

The current phone-camera images are development inputs only. They are useful for
testing image-processing ideas, but cannot provide industrially valid mm values
or PASS/FAIL decisions.

## Current architecture

```text
image loading
    -> controlled preprocessing (gray / R / B / HSV value)
    -> edge detection experiments (Canny / Sobel / Otsu)
    -> geometric measurement in pixels
    -> explicit pixel-to-mm calibration
    -> engineering tolerance check
    -> PASS / FAIL
```

For the stepped cylindrical job, the planned sequence is more specifically:

```text
source image -> locate job -> extract ROI -> normalize/alignment
             -> D1 / D2 / D3 / D4 configured bands -> left/right boundaries
             -> pixel diameters -> calibration -> tolerance decision
```

The layers are intentionally independent:

- `app/core/preprocessor.py` prepares comparable channels without declaring a
  channel inherently best.
- `app/core/edge_detector.py` applies Gaussian/median filtering, Canny, Sobel,
  and Otsu methods and records edge-density and contour-count metrics.
- `app/core/measurement.py` contains pixel-only geometry. For example,
  `PixelDiameterMeasurement(100, 480).diameter_pixels` is `380` pixels.
- `app/core/calibration.py` supplies an explicit calibration interface. The
  current `LinearScaleCalibration` implements `mm = pixels * mm_per_pixel` only
  when that relationship has been demonstrated at the measurement plane.
- `app/core/tolerance_checker.py` checks a calibrated dimension against an
  inclusive symmetric engineering tolerance and returns a structured result.
- `app/core/inspector.py` orchestrates the stages and labels the current image
  runner as an `EXPERIMENT`.
- `app/core/job_detector.py` provides fixed/manual ROIs and a classical
  dark-red silhouette detector for the supplied red-backlit BMP fixture. The
  latter assigns deterministic `job_rNN_cNN` identifiers without manual ROIs.
- `app/core/alignment.py` provides an identity normalizer and a horizontal
  flange-left normalizer that rotates the BMP jobs into the D1-bottom/D4-top
  convention expected by the measurement engine.
- `app/core/measurement_spec.py` holds configurable normalized D1-D4 bands,
  optional nominal dimensions/tolerances, and independent D3/D4 definitions.
- `app/core/diameter_engine.py` converts configurable left/right edge
  localizations into typed per-diameter results.

## Four-diameter concept

The component is a stepped metallic cylinder:

- **D1** is the largest diameter.
- **D2** is smaller than D1.
- **D3** and **D4** are separate physical locations and are expected to be
  approximately equal.

Even if D3 and D4 receive the same nominal diameter and tolerance, the engine
holds them as distinct `DiameterSpec` entries and measures them independently.
That is necessary to identify a local defect in either section.

The vertical positions are intentionally not hard-coded. A `MeasurementRegion`
uses `top_fraction` and `bottom_fraction` of a normalized, aligned job ROI.
They must be set from the engineering drawing and validated fixture geometry.

```python
from app.core.measurement_spec import DiameterSpec, FourDiameterSpec, MeasurementRegion

four_diameter_spec = FourDiameterSpec((
    DiameterSpec("D4", MeasurementRegion(0.00, 0.25)),
    DiameterSpec("D3", MeasurementRegion(0.25, 0.50)),
    DiameterSpec("D2", MeasurementRegion(0.50, 0.75)),
    DiameterSpec("D1", MeasurementRegion(0.75, 1.00)),
))
```

The fractions above illustrate the configuration format only; they are not
measurements or validated locations for the supplied phone images.

`FourDiameterMeasurementEngine.measure_diameter(job_image, diameter_spec)`
uses a replaceable `sobel_x` (default) or `canny` boundary strategy. It reports
left edge, right edge, pixel diameter, optional calibrated mm value, optional
limits, and one of `PASS`, `FAIL`, `NOT_CONFIGURED`, or `MEASUREMENT_FAILED`.
For reflective metal, either strategy needs validation under the final lighting:
specular highlights, shadows, and internal step edges can outscore an outer
boundary if a ROI is not aligned and tightly constrained.

## Calibration and measurement validity

No code path implicitly converts pixels to millimetres. A conversion needs a
specific `CalibrationModel`; the simple linear scale is not sufficient when
perspective, lens distortion, varying working distance, or varying object height
affect the measurement. The interface leaves room for future camera intrinsics,
distortion correction, a calibrated fixture plane, and position-dependent scale.

Before enabling production PASS/FAIL, establish all of the following:

1. The engineering drawing identifies every diameter and its measurement plane.
2. A fixture fixes part pose and working distance, with controlled illumination.
3. Camera/lens calibration, distortion correction, and a traceable reference
   establish measurement uncertainty across the inspection field.
4. Edge localization and geometry fitting are validated with known artefacts.
5. The end-to-end 50-job workload is benchmarked against the one-second target.

The supplied images do not establish the exact diameter locations, nominal
dimensions, tolerance values, reference dimensions, or optical geometry. The
experiment therefore does **not** generate diameter results in mm or fabricate
PASS/FAIL statuses.

## Run the edge experiment

Install dependencies into the project environment:

```bash
.venv/bin/pip install -r requirements.txt
```

Then process all supported images in `data/input`:

```bash
.venv/bin/python scripts/run_edge_experiment.py
```

Or choose paths explicitly:

```bash
.venv/bin/python scripts/run_edge_experiment.py \
  --input data/input \
  --output data/output/edge_experiments
```

For each image, the script saves a six-panel visualization containing original,
grayscale, red, blue, Canny edge, and exploratory contour panels. It also writes
`experiment_report.json`, which compares every configured method using edge
pixel count, edge coverage, contour count, and timings in milliseconds:

- image loading
- preprocessing
- edge detection
- measurement (currently `0.0` because no drawing-defined region exists)
- complete pipeline

Reflective metal and uncontrolled phone lighting can create false edges,
highlights, shadow boundaries, and fragmented contours. Treat edge coverage and
contour count as comparison signals, not accuracy scores. A method should be
selected only after testing in the intended optical configuration.

## Configured four-diameter experiment

When fixture/manual ROIs and D1-D4 bands have been chosen for an experiment,
place them in a JSON file such as:

```json
{
  "job_rois": [{"job_id": "job-01", "x": 100, "y": 120, "width": 160, "height": 280}],
  "diameters": [
    {"name": "D4", "region": {"top_fraction": 0.00, "bottom_fraction": 0.25}},
    {"name": "D3", "region": {"top_fraction": 0.25, "bottom_fraction": 0.50}},
    {"name": "D2", "region": {"top_fraction": 0.50, "bottom_fraction": 0.75}},
    {"name": "D1", "region": {"top_fraction": 0.75, "bottom_fraction": 1.00}}
  ]
}
```

Those values are a schema example only. Replace every ROI and band using the
actual drawing and controlled fixture; do not use the example to inspect parts.

```bash
.venv/bin/python scripts/run_four_diameter_experiment.py data/input/example.jpeg \
  --config path/to/manual_measurement_config.json
```

The script saves an annotated pixel-only image with D1-D4 left/right edges and
pixel distances, plus a structured JSON report. Supplying `nominal_mm` and
`tolerance_mm` together enables a tolerance decision only when a calibration is
also supplied programmatically. Without either prerequisite, output remains
`NOT_CONFIGURED` rather than fabricating a decision.

## Automatic BMP fixture inspection

The four supplied BMP images can be localized and measured without manual ROIs:

```bash
.venv/bin/python scripts/run_bmp_inspection.py \
  --input data/input \
  --output data/output/bmp_inspection \
  --benchmark-runs 10
```

For each BMP, the runner writes an annotated full image, a per-job JSON result,
and a four-rows-per-job CSV result. It also writes `benchmark.json`. With no
measurement configuration, values remain outer-silhouette pixel measurements
and every diameter/job status is `NOT_CONFIGURED`.

To enable millimetres and tolerance decisions, pass an explicit configuration:

```bash
.venv/bin/python scripts/run_bmp_inspection.py \
  --input data/input \
  --output data/output/bmp_inspection \
  --measurement-config data/config/bmp_measurement_config.synthetic.example.json
```

The supplied config file is a **synthetic example only**. Its scale, nominals,
and tolerances must never be used for the supplied BMPs or production. Replace
all of them with a validated calibration and drawing-approved values. The JSON
configuration supports a positive `calibration.mm_per_pixel` and optional
`diameters` entries containing `name`, `nominal_mm`, and `tolerance_mm`.

With calibration alone, `diameter_mm` is emitted but the individual and overall
job statuses remain `NOT_CONFIGURED`. With calibration and a dimension's
tolerance, that diameter becomes `PASS` or `FAIL` at inclusive boundaries. A
job can be `PASS` only when all four dimensions are configured and pass.
Overall job precedence is `MEASUREMENT_FAILED`, `NOT_CONFIGURED`, `FAIL`, then
`PASS`. JSON includes job/image status counts; CSV includes pixel, mm, nominal,
tolerance, limits, diameter status, and job status.

## Example downstream API

Once a drawing-defined pair of localized edges, calibration, and tolerance are
available, downstream stages can be invoked explicitly:

```python
from app.core.calibration import LinearScaleCalibration
from app.core.inspector import DiameterInspectionEngine
from app.core.measurement import PixelDiameterMeasurement
from app.core.tolerance_checker import SymmetricTolerance

pixel_measurement = PixelDiameterMeasurement(left_edge_x=100, right_edge_x=480)
calibration = LinearScaleCalibration(mm_per_pixel=0.052, reference_description="validated fixture plane")
specification = SymmetricTolerance(nominal_mm=20.00, plus_minus_mm=0.05)
calibrated, decision = DiameterInspectionEngine.evaluate_diameter(
    pixel_measurement, calibration, specification
)
print(calibrated.diameter_mm, decision.as_dict())
```

This example demonstrates software flow only; its values are not a valid
calibration or inspection result.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The test suite covers pixel diameter arithmetic, subpixel coordinates,
calibration conversion, invalid inputs, inclusive tolerance boundaries, and
structured tolerance results. It also creates a synthetic stepped job to test
independent D1/D2/D3/D4 bands, matching D3/D4 nominals, missing configuration,
and invalid regions.
