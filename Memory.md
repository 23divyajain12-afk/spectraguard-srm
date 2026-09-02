# Memory.md

## Project Identity

**Working name:** SpectraGuard-SRM

**Target competition:** SIH26142

**Primary objective:** Build a practical, low/zero-training multispectral super-resolution and spectral-unmixing prototype for Sentinel-2 imagery.

## Strategic Decision

Do not train a spatial-spectral super-resolution network from scratch during the 3–4 day build.

Instead:

1. Use a pretrained SR model as a spatial-detail prior.
2. Keep the original multispectral data as the spectral base.
3. Inject learned spatial detail analytically into the spectral bands.
4. Enforce consistency by degrading the output back toward the original Sentinel observation.
5. Quantify spectral/geospatial error and uncertainty.
6. Perform sensor-aware spectral unmixing with FCLS.

---

## Core Design Philosophy

### Neural network role

The neural network is **not** trusted to generate the final multispectral cube directly.

Its role is:

> Estimate plausible high-frequency spatial structure.

### Analytical role

Analytical methods decide:

- how much spatial detail enters each spectral band,
- whether the reconstructed image remains consistent with the observed Sentinel data,
- how endmembers are represented in Sentinel's spectral space,
- and how material/land-cover fractions are estimated.

This makes the system more explainable and less dependent on training.

---

## Corrected Technical Position

### PCA

PCA can be used to decorrelate spectral dimensions and help construct a compact spatial representation, but:

- PC1 is not inherently "the spatial component".
- PC1 is not guaranteed to contain all structural information.
- Replacing a PCA component changes the reconstructed image.

Therefore:

> PCA is a useful representation/feature transform, not a proof of spectral preservation.

### Spectral preservation

The system should claim **spectral consistency / controlled spectral deviation**, not absolute spectral invariance.

The strongest defensible mechanism is:

1. Start from bicubic multispectral bands.
2. Add a controlled spatial residual.
3. Downsample the resulting HR cube to the observation scale.
4. Compare against the original Sentinel observation.
5. Apply an analytical correction.
6. Report SAM and reconstruction error.

---

## Core Mathematical Path

For each spectral band `k`:

`B_k = bicubic(I_k)`

Generate a spatial residual:

`D = SR(spatial_input) - bicubic(spatial_input)`

Then:

`HR_k = B_k + alpha_k * D`

with a bandwise coefficient such as:

`alpha_k = Cov(B_k, B_spatial) / (Var(B_spatial) + epsilon)`

followed by clipping or regularization as justified by validation.

Then apply consistency projection:

`HR -> D(HR) -> compare with I_original -> correction`

The exact degradation model must be logged.

---

## Spectral Unmixing Memory

Never use plain NNLS and call it FCLS.

FCLS requires:

`A s ≈ x`

with:

`s_i >= 0`

and:

`sum_i s_i = 1`

The implementation may use:

- a dedicated FCLS solver,
- a constrained optimizer,
- or a mathematically equivalent formulation.

---

## Endmember Memory

The spectral library workflow must be:

`USGS spectrum`
`↓`
`Sentinel-2 spectral response function`
`↓`
`band integration`
`↓`
`13-band Sentinel-compatible endmember`
`↓`
`FCLS`

Do not directly compare high-resolution laboratory spectra to Sentinel band vectors.

---

## Data Memory

Primary observation source:

- Sentinel-2 multispectral imagery, preferably Level-2A reflectance products.

Expected considerations:

- native band resolutions differ,
- all bands must be placed on a consistent working grid,
- cloud and shadow pixels must be masked or flagged,
- CRS and affine metadata must be preserved.

Reference data:

- higher-resolution imagery where legally and practically available,
- or suitable reference/benchmark scenes for validation.

Spectral library:

- USGS Spectral Library or another documented spectral library.

Sensor data:

- Sentinel-2 spectral response functions.

---

## Product Outputs

The system should produce:

1. High-resolution multispectral raster.
2. Bicubic baseline.
3. Spectral reconstruction metrics.
4. Spectral angle map.
5. Uncertainty/confidence map.
6. Optional abundance maps.
7. FCLS abundance fractions.
8. Processing metadata.
9. Validation report.

---

## Demo Memory

The strongest demonstration is comparative:

`Original 10 m | Bicubic 4x | Proposed`

Then show:

- a zoomed spatial feature,
- spectrum at a selected location,
- SAM/reconstruction error,
- uncertainty overlay,
- FCLS abundance map.

The narrative should be:

> The system uses a learned spatial prior, but the measured Sentinel observation remains the anchor for spectral and geometric accountability.

---

## Jury Memory

### Devil's judge

Likely concerns:

- hallucinated details,
- false spectral-preservation claims,
- wrong FCLS definition,
- lack of validation,
- geospatial misalignment,
- mismatch between claimed and actual resolution.

### Field expert

Likely concerns:

- Sentinel-2 band resolutions,
- reflectance preprocessing,
- sensor spectral response,
- endmember compatibility,
- SAM/ERGAS interpretation,
- whether output is scientifically meaningful.

### Innovation promoter

Needs:

- a clear separation of pretrained learning from analytical constraint,
- a replaceable SR backbone,
- an explicit consistency-projection mechanism,
- uncertainty estimation,
- useful downstream unmixing.

### General judge

Needs:

- obvious before/after imagery,
- intuitive explanation,
- a working click-to-inspect demo,
- clear application value.

---

## Time Constraint Memory

Target implementation window:

**3–4 days**

Priority order:

`Data -> SR integration -> Fusion -> Consistency -> Validation -> FCLS -> UI`

Do not prioritize:

- training a new model,
- large-scale hyperparameter search,
- elaborate distributed inference,
- unnecessary microservices,
- overly ambitious web infrastructure.

---

## Non-Negotiable Benchmark

Bicubic interpolation must remain in the pipeline.

The proposed method must demonstrate at least one meaningful improvement over bicubic in:

- visual spatial quality,
- metric performance under an appropriate reference,
- or downstream application quality,

while transparently reporting any trade-offs.

---

## Known Risk

A pretrained natural-image SR network can generate textures that are visually plausible but physically unsupported.

Mitigation:

- use it primarily as a spatial prior,
- limit detail injection,
- consistency-project against original observations,
- calculate uncertainty,
- explicitly label inferred details.

---

## Current Status

This project file records the intended architecture and engineering constraints. It does not imply that every module has already been implemented or validated.
