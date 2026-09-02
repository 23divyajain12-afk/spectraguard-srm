# Skills.md

## Purpose

This file defines the skills, working rules, and engineering responsibilities required to build the SIH26142 prototype described in the project architecture.

## Project Objective

Build a low/zero-training Sentinel-2 multispectral super-resolution and spectral-unmixing system that:

1. Starts from Sentinel-2 L2A multispectral imagery.
2. Produces a spatially enhanced multispectral product targeting `<4 m` output resolution where the selected pretrained model supports it.
3. Preserves spectral and geospatial consistency through analytical constraints rather than training a new SR network from scratch.
4. Quantifies reconstruction quality and uncertainty.
5. Converts spectral-library endmembers into a Sentinel-compatible representation.
6. Produces abundance maps using Fully Constrained Least Squares (FCLS).
7. Provides an interactive, explainable demonstration suitable for SIH judging.

---

## Core Engineering Skills

### 1. Remote Sensing / Raster Processing

The implementation agent must be comfortable with:

- Sentinel-2 L1C/L2A conventions.
- Band-specific native resolutions.
- Reflectance scaling.
- Cloud and cloud-shadow masking.
- CRS and reprojection.
- Affine transforms and georeferencing.
- Raster windows/chunking.
- GeoTIFF/COG/JP2 handling where applicable.
- Spatial resampling and alignment.
- Nodata propagation.

Primary Python ecosystem:

- `rasterio`
- `numpy`
- `scipy`
- `pyproj`
- `xarray` where useful
- `dask` only when genuinely necessary

### 2. Multispectral Mathematics

The agent must understand:

- PCA and inverse PCA.
- Covariance and correlation.
- Spectral Angle Mapper (SAM).
- Euclidean / RMSE-style reconstruction errors.
- Spectral normalization.
- Bandwise covariance-based detail injection.
- Constrained least squares.
- Non-negative least squares versus true FCLS.
- Sensor-response-function based spectral resampling.

Important rule:

**Do not claim that PCA or inverse PCA alone mathematically guarantees spectral preservation after modifying PCA coefficients.**

### 3. Super-Resolution

The agent must be able to:

- Integrate pretrained SR models.
- Separate model inference from multispectral fusion.
- Work with tiled inference.
- Handle overlap and stitching.
- Keep the SR model replaceable.
- Compare at least one strong pretrained model against bicubic interpolation.
- Record the model, weights, scale, input preprocessing, and output assumptions.

Preferred philosophy:

> The neural model estimates spatial detail; analytical constraints decide how that detail may enter the multispectral product.

### 4. Spectral-Spatial Fusion

The core fusion skill is implementing a detail-residual formulation such as:

`HR_k = B_k + alpha_k * D`

where:

- `B_k` = bicubic-upscaled band `k`
- `D` = spatial high-frequency/detail residual from the pretrained SR branch
- `alpha_k` = analytically estimated per-band injection coefficient

A practical first estimator is covariance-based:

`alpha_k = Cov(B_k, B_spatial) / (Var(B_spatial) + epsilon)`

This is a starting point, not a sacred formula. The implementation must validate whether the choice improves quality and does not destabilize spectra.

### 5. Consistency Projection

The agent must implement a degradation/reprojection loop:

`HR -> degrade to Sentinel resolution -> compare with original -> correct HR`

A baseline update can be:

`HR_(t+1) = HR_t + lambda * U(I_original - D(HR_t))`

where:

- `D` = degradation/downsampling operator approximating Sentinel observation formation
- `U` = compatible upsampling operator
- `lambda` = correction step

The exact operator must be documented.

### 6. Spectral Unmixing

The agent must understand the distinction:

- NNLS: non-negativity only.
- FCLS: non-negativity + sum-to-one constraint.

The desired optimization is:

`min_s ||x - A s||^2`

subject to:

`s_i >= 0`

and

`sum_i s_i = 1`

The implementation must not falsely label plain `scipy.optimize.nnls` as FCLS.

### 7. Spectral Library Handling

The agent must understand that a high-resolution USGS spectrum should not be compared directly with a Sentinel multispectral pixel.

Preferred pipeline:

`USGS spectrum -> Sentinel-2 spectral response integration -> Sentinel-compatible endmember -> FCLS`

The spectral-response data source and interpolation/integration method must be recorded.

### 8. Validation

Required validation concepts:

- PSNR where a suitable reference exists.
- SSIM where appropriate.
- SAM.
- ERGAS.
- Per-band RMSE.
- Downsampling/reconstruction error.
- Geospatial alignment checks.
- Uncertainty/confidence maps.

When true high-resolution ground truth is unavailable, the system must clearly label results as proxy, internal-consistency, or reference-based validation rather than pretending they are direct ground-truth measurements.

### 9. Software Engineering

The agent should use:

- Modular Python packages.
- Type hints for public functions.
- Config-driven parameters.
- Structured logging.
- Deterministic seeds where stochastic behavior exists.
- Unit tests for mathematical functions.
- Small test fixtures before full-tile inference.
- Clear CLI entry points.
- Version-pinned dependencies when practical.

### 10. Visualization / Demo Engineering

The final demo should support:

- Original Sentinel-2 view.
- Bicubic baseline.
- SR/fused output.
- Spectral curve comparison.
- SAM / reconstruction metrics.
- Uncertainty overlay.
- FCLS abundance layers.
- AOI/click inspection.
- Metadata display.

The UI should explain what is observed versus inferred.

---

## Agent Working Rules

### Rule 1 — Do not invent scientific guarantees

Avoid phrases such as:

- "mathematically guaranteed spectral preservation"
- "all spatial information is in PC1"
- "true ground truth" when only proxy reference data exists
- "FCLS" when the implementation only performs NNLS

### Rule 2 — Preserve geospatial metadata

Every raster-producing step must explicitly consider:

- CRS
- transform
- dimensions
- resolution
- bounds
- nodata
- band order

### Rule 3 — Keep the pretrained model replaceable

The architecture should allow:

- Real-ESRGAN-style models
- RCAN
- EDSR
- SwinIR
- Sentinel-specific pretrained SR models

without rewriting the spectral pipeline.

### Rule 4 — Never process an entire satellite tile unnecessarily

Use:

- AOI crops
- windows
- patches
- tiled inference
- caching

for development.

### Rule 5 — Benchmark every enhancement against bicubic

The system must always retain a simple baseline:

`Bicubic 4x`

Any complexity is justified only if the output or validation metrics improve relative to the baseline.

### Rule 6 — Make uncertainty visible

The uncertainty system should never be hidden from the user. A key product is:

`high-resolution result + confidence / uncertainty`

### Rule 7 — Prefer a robust 80% prototype over a fragile 100% research system

For a 3–4 day SIH build:

1. Make data ingestion reliable.
2. Make one pretrained SR route work.
3. Implement spectral-aware fusion.
4. Implement consistency correction.
5. Implement validation.
6. Implement FCLS.
7. Polish the demo.

Do not spend the entire build window tuning a novel deep-learning model.

---

## Suggested Repository Skills Map

```text
src/
  data/
    sentinel_io.py
    preprocessing.py
    geospatial.py

  sr/
    model_adapter.py
    tiling.py
    spatial_representation.py

  fusion/
    bicubic.py
    detail_residual.py
    spectral_injection.py

  consistency/
    degradation.py
    projection.py
    spectral_metrics.py

  library/
    usgs_io.py
    sensor_response.py
    endmember_preparation.py

  unmixing/
    fcls.py

  validation/
    metrics.py
    uncertainty.py

  app/
    dashboard.py

  pipeline/
    orchestrator.py
```

## Definition of Done

A phase is not complete until:

- the code runs on a small local fixture,
- outputs are saved with metadata,
- numerical sanity checks pass,
- failures are logged,
- assumptions are documented,
- and the result can be reproduced from configuration.
