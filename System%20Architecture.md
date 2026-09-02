# System Architecture.md

# SpectraGuard-SRM
## Detailed Modular, Phased System Architecture

---

# 0. Architecture Objective

The system is designed for a 3–4 day SIH prototype and intentionally avoids training a new spatial-spectral super-resolution network from scratch.

The architecture separates the problem into:

1. Observation preparation.
2. Spatial prior generation.
3. Analytical spectral-spatial fusion.
4. Observation-consistency correction.
5. Validation and uncertainty.
6. Sensor-aware spectral unmixing.
7. Demonstration/application.

The central design principle is:

> **Use a pretrained network for spatial detail estimation, and use analytical constraints to keep the generated multispectral product accountable to the measured Sentinel-2 observation.**

---

# 1. High-Level Architecture

```text
                     ┌─────────────────────────┐
                     │ Sentinel-2 L2A          │
                     │ multispectral imagery   │
                     └────────────┬────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ PHASE 1: DATA INGESTION       │
                  │ - read bands                  │
                  │ - scaling / nodata            │
                  │ - metadata                    │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ PHASE 2: GEO-SPATIAL          │
                  │ PREPROCESSING                 │
                  │ - CRS / grid alignment       │
                  │ - resampling                  │
                  │ - masks                       │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                       Multispectral 10m
                              cube
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
      ┌───────────────────┐             ┌────────────────────┐
      │ Spectral Base     │             │ Spatial Branch     │
      │ Bicubic Upscaling │             │ representation     │
      └─────────┬─────────┘             └─────────┬──────────┘
                │                                  │
                │                                  ▼
                │                         Pretrained SR
                │                                  │
                │                                  ▼
                │                         HR spatial estimate
                │                                  │
                │                                  ▼
                │                         High-frequency D
                │                                  │
                └────────────────┬─────────────────┘
                                 ▼
                  ┌───────────────────────────────┐
                  │ PHASE 3: SPECTRAL-SPATIAL     │
                  │ FUSION                        │
                  │ HR_k = B_k + alpha_k D       │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ PHASE 4: CONSISTENCY          │
                  │ PROJECTION                    │
                  │ HR -> D(HR) -> compare       │
                  │ -> correction                │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                         <4m Target Cube
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
               Metrics       Uncertainty      FCLS
               /Validation   Map              Unmixing
                  │              │              │
                  └──────┬───────┴──────┬───────┘
                         ▼              ▼
                    Validation      Application
                     report         /Dashboard
```

---

# 2. Architectural Principles

## P1 — Modular SR Backbone

The SR model is an adapter, not the architecture itself.

The following can be swapped:

- Real-ESRGAN-style model
- RCAN
- EDSR
- SwinIR
- Sentinel-specific pretrained model

without rewriting:

- fusion,
- consistency,
- validation,
- FCLS.

---

## P2 — Observation Remains the Anchor

The original Sentinel observation is never discarded.

The enhanced image must remain connected to:

`original observation -> enhancement -> simulated observation`

This prevents the system from becoming a pure image-generation pipeline.

---

## P3 — Explicit Inference Versus Measurement

The architecture must distinguish:

**Measured**

- original Sentinel observations.

**Interpolated**

- bicubic base.

**Model-inferred**

- SR spatial detail.

**Analytically constrained**

- final fused cube.

**Estimated**

- FCLS abundances.

This distinction is important for scientific credibility.

---

# 3. Phase 0 — Project Setup and Configuration

## Goal

Create the reproducible skeleton before implementation.

## Inputs

- model choice
- scale factor
- selected Sentinel bands
- AOI
- paths
- validation mode

## Outputs

- configuration file
- environment
- repository structure
- logging setup

## Suggested configuration

```yaml
project:
  name: SpectraGuard-SRM

data:
  bands: [...]
  target_resolution_m: 2.5
  working_resolution_m: 10

sr:
  model: <pretrained_model>
  scale: 4
  tile_size: 256
  overlap: 32

fusion:
  epsilon: 1e-8
  alpha_min: 0.0
  alpha_max: 1.5

consistency:
  iterations: 3
  lambda: 0.5

unmixing:
  solver: fcls
  endmembers: [...]
```

## Exit Criterion

Running the project with a sample configuration starts successfully and reports versions/configuration.

---

# 4. Phase 1 — Data Ingestion

## Goal

Create a clean Sentinel multispectral cube.

## Module

`src/data/sentinel_io.py`

## Responsibilities

- locate bands,
- read raster data,
- apply scale factors,
- preserve nodata,
- expose metadata.

## Output

```text
SentinelCube
  data: C x H x W
  CRS
  transform
  resolution
  bounds
  mask
  band_names
```

## Tests

- band count,
- dimensions,
- finite values,
- metadata existence.

---

# 5. Phase 2 — Geospatial and Radiometric Preprocessing

## Goal

Convert heterogeneous Sentinel-2 bands into a consistent analysis grid.

## Module

`src/data/preprocessing.py`

## Operations

1. Cloud/shadow masking.
2. Nodata handling.
3. Reprojection if needed.
4. Resampling lower-resolution bands.
5. Aligning all bands.
6. Reflectance normalization.
7. AOI cropping.

## Critical Invariant

All output bands share:

- same dimensions,
- same transform,
- same CRS,
- same pixel grid.

## Output

`I_10m`

conceptually:

`I_10m ∈ R^(H × W × C)`

---

# 6. Phase 3 — Baseline Construction

## Goal

Create a baseline that all later methods must beat or at least justify.

## Module

`src/fusion/bicubic.py`

For each band:

`B_k = Bicubic(I_k, scale=4)`

## Why it exists

Bicubic is:

- fast,
- deterministic,
- easy to explain,
- scientifically useful as a control.

## Output

`B ∈ R^(4H × 4W × C)`

or another configured target dimension.

---

# 7. Phase 4 — Spatial Representation

## Goal

Create the input used by the pretrained SR model.

## Options

### Option A — Pseudo-RGB / selected-band representation

Simple and robust for a natural-image SR model.

### Option B — PCA representation

Transform:

`X -> PCA(X)`

Use PCA features as a compact spatial representation.

Important:

PCA is a feature transform only. Do not claim PC1 contains all spatial information.

### Option C — Sentinel-native pretrained model

Preferred if a high-quality Sentinel-specific pretrained model is available and easy to integrate.

## Selection Rule

For a 3–4 day prototype:

1. Sentinel-specific pretrained model if it runs reliably.
2. Otherwise a stable pretrained general SR model.
3. PCA-based branch only where it demonstrably helps.

---

# 8. Phase 5 — Pretrained SR Inference

## Goal

Generate high-resolution spatial structure without custom training.

## Module

`src/sr/model_adapter.py`

## Responsibilities

- load pretrained weights,
- normalize input,
- perform tiled inference,
- stitch overlapping tiles,
- invert normalization,
- return spatial estimate.

## Tiling

For large images:

```text
large image
    ↓
overlapping tiles
    ↓
SR inference per tile
    ↓
weighted/valid-region stitch
    ↓
full-resolution spatial estimate
```

## Output

`S_SR`

---

# 9. Phase 6 — Spatial Residual Extraction

## Goal

Extract only the spatial information that the SR model added beyond bicubic.

## Module

`src/fusion/detail_residual.py`

Construct:

`S_base = Bicubic(S_input)`

Then:

`D = S_SR - S_base`

`D` is the learned high-frequency spatial residual.

## Why residual form

This avoids treating the model output as the final multispectral truth.

The system only asks:

> What spatial detail did the pretrained model add?

---

# 10. Phase 7 — Spectral-Aware Detail Injection

## Goal

Transfer spatial detail into every spectral band under analytical control.

## Module

`src/fusion/spectral_injection.py`

For each band `k`:

`HR_k = B_k + alpha_k D`

A baseline coefficient:

`alpha_k = Cov(B_k, B_spatial) / (Var(B_spatial) + epsilon)`

## Controls

- clip `alpha_k`,
- prevent invalid reflectance,
- preserve nodata,
- optionally apply spatial smoothing to coefficient maps,
- document any band-specific constraints.

## Important

The coefficient formula is a starting design.

The implementation should compare:

- fixed alpha,
- global alpha,
- local alpha,

and retain the simplest validated option.

---

# 11. Phase 8 — Spectral Safety Checks

## Goal

Catch physically suspicious output before consistency projection.

## Checks

For each pixel/band:

- finite value?
- within configured reflectance range?
- extreme deviation from bicubic?
- excessive spectral angle?
- nodata contamination?

## Outputs

- cleaned HR cube,
- preliminary SAM map,
- anomaly mask.

---

# 12. Phase 9 — Observation Degradation Model

## Goal

Approximate how the high-resolution result would appear at Sentinel scale.

## Module

`src/consistency/degradation.py`

Conceptually:

`I_pred = D(HR)`

where `D` includes:

- appropriate spatial blur or point-spread approximation,
- downsampling,
- band/channel handling as appropriate.

A first prototype may use a documented approximation rather than a physically exact Sentinel instrument model.

---

# 13. Phase 10 — Consistency Projection

## Goal

Force the generated high-resolution cube to remain consistent with the original observation.

## Module

`src/consistency/projection.py`

For iteration `t`:

`R_t = I_original - D(HR_t)`

Then:

`HR_(t+1) = HR_t + lambda U(R_t)`

where:

- `U` = compatible upsampling,
- `lambda` = correction factor.

## Optional refinement

Weight correction using:

- valid pixel masks,
- band reliability,
- local confidence.

## Exit Criterion

The consistency residual decreases or remains stable without degrading the enhancement beyond acceptable thresholds.

---

# 14. Phase 11 — Spectral Validation

## Goal

Measure whether the enhancement changes spectra excessively.

## Module

`src/validation/spectral_metrics.py`

Calculate:

### Spectral Angle Mapper

For original/reference-compatible vectors `x` and reconstructed vectors `y`:

`SAM(x,y) = arccos( x·y / (||x|| ||y||) )`

### Other metrics

- per-band RMSE,
- MAE,
- normalized spectral error,
- reconstruction residual.

## Output

- mean SAM,
- percentile SAM,
- spatial SAM map,
- per-band metrics.

---

# 15. Phase 12 — Spatial / Reference Validation

## Goal

Evaluate whether the model truly improves spatial detail when a suitable reference exists.

## Module

`src/validation/metrics.py`

Possible measures:

- PSNR,
- SSIM,
- RMSE,
- ERGAS where assumptions are satisfied.

## Comparison

Always compare:

`Bicubic vs Proposed`

and, where possible:

`Pretrained SR baseline vs Proposed`

## Important

A metric is meaningful only when the reference and prediction are comparable in:

- spatial resolution,
- geometry,
- spectral meaning.

---

# 16. Phase 13 — Uncertainty Estimation

## Goal

Identify areas where the model's inferred detail should be treated cautiously.

## Module

`src/validation/uncertainty.py`

Candidate terms:

`U_SAM`
`U_reconstruction`
`U_detail`

Combine:

`U = w1*U_SAM + w2*U_reconstruction + w3*U_detail`

Normalize to a display range.

## Interpretation

High uncertainty means:

> The output contains inferred structure for which the evidence from the measured observation is weak or inconsistent.

This is not automatically a calibrated probability.

---

# 17. Phase 14 — Sensor-Aware Spectral Endmembers

## Goal

Prepare the spectral library for Sentinel-compatible unmixing.

## Modules

```text
src/library/usgs_io.py
src/library/sensor_response.py
src/library/endmember_preparation.py
```

## Pipeline

```text
USGS spectrum
      ↓
clean missing values
      ↓
interpolate onto sensor-response wavelength grid
      ↓
integrate through Sentinel-2 RSR
      ↓
Sentinel-compatible endmember
```

Output:

`A ∈ R^(C × M)`

---

# 18. Phase 15 — FCLS Unmixing

## Goal

Estimate material/land-cover fractions for each HR pixel.

## Module

`src/unmixing/fcls.py`

Optimization:

`min ||x - A s||²`

subject to:

`s_i >= 0`

`sum_i s_i = 1`

## Output

For each selected endmember:

- abundance raster,
- optional residual/error raster.

Example:

```text
Vegetation = 0.81
Soil       = 0.15
Other      = 0.04
```

The exact number of endmembers must be determined by the selected scene and library.

---

# 19. Phase 16 — Application Layer

## Goal

Turn the numerical pipeline into something a judge can understand.

## Minimum demo

```text
Original Sentinel
       │
       ├── Bicubic
       │
       └── Proposed
              │
              ├── Spectral inspector
              ├── Uncertainty
              └── FCLS abundance
```

## Recommended interaction

User selects a pixel/region.

Dashboard shows:

- original spectrum,
- reconstructed spectrum,
- SAM,
- confidence/uncertainty,
- abundance fractions.

---

# 20. Phase 17 — Final Dashboard

## Module

`src/app/dashboard.py`

Possible technology:

- Streamlit for speed,
- or a minimal React/MapLibre frontend only if necessary.

For a 3–4 day hackathon build, prefer Streamlit or another lightweight Python UI.

## Required views

### View 1 — Spatial comparison

- Original
- Bicubic
- Proposed

### View 2 — Spectral chart

At selected point:

- measured spectrum,
- fused spectrum,
- reference spectrum when available.

### View 3 — Uncertainty

Interactive uncertainty overlay.

### View 4 — Unmixing

Abundance maps and fractions.

### View 5 — Metrics

- SAM
- RMSE
- PSNR/SSIM where applicable
- consistency error

---

# 21. Phase 18 — Packaging and Reproducibility

## Goal

Make the prototype runnable by another team member.

## Required

- `requirements.txt` or equivalent
- configuration file
- README
- download/setup instructions
- sample data path
- model checkpoint instructions
- one-command or one-notebook demo

## Metadata

Every run should record:

```text
scene
AOI
bands
model
weights
scale
fusion parameters
consistency parameters
endmembers
metrics
timestamp
software versions
```

---

# 22. Modular Interfaces

## Interface A — Data

```python
cube = load_sentinel(scene_path, config)
cube = preprocess(cube, config)
```

## Interface B — SR

```python
spatial_hr = sr_model.predict(spatial_input)
```

## Interface C — Fusion

```python
hr_cube = fuse_multispectral(
    bicubic_cube,
    spatial_base,
    spatial_hr,
    config
)
```

## Interface D — Consistency

```python
hr_cube = project_to_measurement_consistency(
    hr_cube,
    original_cube,
    config
)
```

## Interface E — Validation

```python
metrics = evaluate(hr_cube, original_cube, reference, config)
uncertainty = estimate_uncertainty(hr_cube, metrics, config)
```

## Interface F — Unmixing

```python
endmembers = prepare_endmembers(library, sensor_response, config)
abundance = fcls(hr_cube, endmembers, config)
```

---

# 23. Phase Dependency Graph

```text
P0 Setup
  ↓
P1 Ingestion
  ↓
P2 Geospatial Preprocessing
  ↓
P3 Bicubic Baseline
  ↓
P4 Spatial Representation
  ↓
P5 Pretrained SR
  ↓
P6 Residual Extraction
  ↓
P7 Spectral Fusion
  ↓
P8 Safety Checks
  ↓
P9 Degradation Model
  ↓
P10 Consistency Projection
  ↓
P11 Spectral Validation
  ├───────────────┐
  ▼               ▼
P12 Reference     P13 Uncertainty
Validation             │
  │                    │
  └─────────┬──────────┘
            ▼
P14 Sensor-aware Endmembers
            ↓
P15 FCLS
            ↓
P16 Application
            ↓
P17 Dashboard
            ↓
P18 Packaging
```

---

# 24. 3-Day Execution Plan

## Day 1 — Data + SR

### Milestone A

- Sentinel loader works.
- Bands aligned.
- Small AOI saved.

### Milestone B

- Bicubic baseline works.
- Pretrained SR model runs on spatial representation.

### Milestone C

- Tiled inference works.

---

# 25. Day 2 — Differentiating Algorithm

## Milestone D

Implement:

`D = SR - Bicubic`

Implement:

`HR_k = B_k + alpha_k D`

Implement consistency degradation.

## Milestone E

Implement consistency projection.

## Milestone F

Compute:

- SAM,
- reconstruction error,
- basic uncertainty.

At the end of Day 2, the team should have the **core research contribution** working.

---

# 26. Day 3 — Unmixing + Demo

## Milestone G

- USGS spectra loaded.
- Sentinel sensor response applied.
- FCLS working.

## Milestone H

- abundance maps generated.

## Milestone I

- dashboard with side-by-side comparison,
- spectrum inspector,
- uncertainty,
- abundance map.

---

# 27. Day 4 — Buffer / Polish

Use only if available.

Focus on:

- bugs,
- speed,
- visual polish,
- additional validation,
- stronger scene,
- pitch deck,
- reproducibility.

Do not start an entirely new model architecture on Day 4.

---

# 28. Failure-Containment Strategy

The architecture should support fallbacks.

## Failure 1 — Preferred SR model does not run

Fallback:

`second pretrained SR model`

## Failure 2 — FCLS solver unstable

Fallback:

- smaller endmember set,
- better-conditioned endmembers,
- constrained optimizer.

## Failure 3 — Reference data unavailable

Fallback:

- synthetic degradation benchmark,
- internal consistency validation,
- transparent labeling.

## Failure 4 — Full-scene processing too slow

Fallback:

- smaller AOI,
- fewer tiles,
- cached inference,
- GPU execution.

## Failure 5 — Detail injection creates artifacts

Fallback:

- lower alpha,
- global coefficient,
- stronger consistency correction,
- uncertainty flagging.

---

# 29. Minimum Viable Prototype

The absolute minimum credible system is:

```text
Sentinel-2
   ↓
Preprocess
   ↓
Bicubic baseline ─────────────┐
                              │
Spatial representation        │
   ↓                          │
Pretrained SR                 │
   ↓                          │
Spatial residual              │
   ↓                          │
Spectral-aware fusion ◄───────┘
   ↓
Consistency correction
   ↓
<4m target cube
   ↓
SAM + reconstruction error
   ↓
uncertainty
   ↓
sensor-adjusted endmembers
   ↓
FCLS
   ↓
dashboard
```

---

# 30. Innovation Statement

The system should not claim:

> "We invented a new deep-learning super-resolution network."

It should claim:

> **"We designed a constraint-driven multispectral super-resolution pipeline that separates learned spatial enhancement from spectral accountability. A pretrained SR model estimates spatial detail, analytical band-aware fusion controls its injection, and measurement-consistency projection forces the final high-resolution cube to remain compatible with the original Sentinel observation. Sensor-aware FCLS then converts the resulting spectra into interpretable abundance maps."**

---

# 31. What Makes the Architecture Defensible

### Scientifically

- original observation retained,
- spectral deviation measured,
- geospatial metadata retained,
- sensor-response-aware endmembers,
- explicit FCLS constraints,
- uncertainty exposed.

### Engineering

- pretrained models,
- modular interfaces,
- replaceable SR backbone,
- tiled inference,
- simple mathematical fusion,
- lightweight deployment.

### Competition

- visible before/after,
- measurable improvement,
- clear innovation,
- application output,
- understandable explanation for non-specialists.

---

# 32. Important Claims to Avoid

Never state:

1. "PC1 contains all spatial information."
2. "Inverse PCA mathematically guarantees spectral preservation."
3. "NNLS is FCLS."
4. "The model produces ground truth."
5. "Every generated 2.5–4 m pixel is physically observed."
6. "A higher-resolution-looking image automatically has higher scientific accuracy."

Preferred language:

- spatial prior,
- inferred detail,
- spectral consistency,
- observation-consistent reconstruction,
- reference-based validation,
- uncertainty-aware output.

---

# 33. Final Architecture Summary

The complete system is:

```text
MEASURED DATA
     │
     ▼
Sentinel-2 preprocessing
     │
     ├──────────────► Spectral base ──────────────┐
     │                                             │
     ▼                                             │
Spatial representation                            │
     │                                             │
     ▼                                             │
Pretrained SR                                      │
     │                                             │
     ▼                                             │
Spatial residual                                   │
     │                                             │
     └────────────────► Analytical fusion ◄────────┘
                              │
                              ▼
                    Consistency projection
                              │
                              ▼
                       Enhanced cube
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
         Validation       Uncertainty        FCLS
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                         Dashboard
```

The architecture is deliberately designed so that the **pretrained model is replaceable, while the spectral-accountability layer remains the project's core contribution**.
