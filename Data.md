# Data.md

## 1. Data Strategy

The system uses four major data classes:

1. **Observation data** — Sentinel-2 imagery.
2. **Reference data** — higher-resolution imagery or benchmark data.
3. **Spectral-library data** — material/land-cover endmembers.
4. **Sensor metadata** — Sentinel-2 spectral response and acquisition metadata.

The architecture must keep these classes separate because they serve different purposes.

---

# 2. Sentinel-2 Observation Data

## Preferred Product

Use Sentinel-2 Level-2A surface reflectance imagery for the primary pipeline.

### Required information

At minimum:

- spectral band values,
- acquisition timestamp,
- CRS,
- affine transform,
- pixel dimensions,
- native resolution,
- scene bounds,
- nodata/mask information,
- cloud/shadow quality information where available.

## Spectral Bands

The implementation should preserve the complete multispectral band set used by the project.

The working spectral cube is conceptually:

`I in R^(H x W x C)`

where `C` is the selected Sentinel-2 band count.

The architecture may use the full set or a documented subset, but the chosen band list must be explicit in configuration.

---

# 3. Native Resolution Issue

Sentinel-2 bands are not all natively sampled at the same ground sampling distance.

Typical resolution groups include:

- 10 m bands,
- 20 m bands,
- 60 m bands.

Therefore the preprocessing stage must explicitly resample lower-resolution bands onto a common working grid.

Do not describe this as "all original bands are natively 10 m."

Instead:

> The system constructs a common Sentinel-2 analysis grid, then resamples bands according to a documented method.

The resampling method must be stored in the run metadata.

---

# 4. Reflectance and Scaling

The loader must detect and correctly handle:

- integer-scaled surface reflectance,
- floating reflectance,
- nodata values,
- fill values,
- masks.

Internally the pipeline should use a documented floating-point representation.

Do not silently assume all downloaded products use the same numerical scale.

---

# 5. Cloud and Shadow Masking

The pipeline should create an explicit quality mask.

At minimum, the system should be capable of flagging/removing:

- clouds,
- cloud shadows,
- unusable pixels,
- nodata.

For a rapid prototype, cloud masking can be simplified, but the simplification must be visible in metadata.

---

# 6. Geospatial Contract

Every raster entering or leaving a pipeline module must carry:

- CRS
- transform
- width
- height
- resolution
- bounds
- nodata
- band names/order

Critical invariant:

> A pixel coordinate must map consistently between the original observation, processed cube, enhanced cube, validation data, and abundance maps.

---

# 7. AOI / Tile Strategy

Development should not start with a huge full-scene product.

Preferred flow:

1. Select a small AOI.
2. Crop all data consistently.
3. Test preprocessing.
4. Run SR.
5. Validate.
6. Scale to larger windows.

For large scenes:

- tile the scene,
- use overlap during SR,
- stitch tiles,
- remove seam artifacts,
- preserve original geospatial indexing.

---

# 8. Reference Data

Reference data is used to estimate how close the enhanced product is to a finer-resolution observation.

Possible classes:

- PlanetScope or similar commercial imagery where access is available and permitted.
- Public benchmark datasets.
- High-resolution aerial/satellite imagery for selected scenes.
- Synthetic degradation tests where a high-resolution source can be deliberately degraded to simulate Sentinel-like observations.

## Important terminology

Do not call every reference dataset "ground truth."

Use:

- reference imagery,
- higher-resolution reference,
- proxy ground truth,
- synthetic benchmark,

as appropriate.

---

# 9. Validation Data Contract

Validation data must be:

- spatially aligned,
- temporally compatible where possible,
- spectrally comparable where possible,
- documented with acquisition and preprocessing details.

If the reference has different spectral characteristics, use an appropriate comparison space rather than pretending the bands are identical.

---

# 10. Spectral Library

## Candidate Source

USGS Spectral Library or another documented spectral library.

## Required fields

For each endmember:

- name
- material/land-cover class
- wavelength vector
- reflectance vector
- source metadata
- measurement conditions if available

## Processing Requirement

Do not directly compare arbitrary high-resolution spectra to Sentinel vectors.

Use:

`library spectrum`
`-> Sentinel-2 spectral response function`
`-> band integration`
`-> Sentinel-compatible endmember`

Conceptually, for Sentinel band `k`:

`e_k = integral(r(lambda) * RSR_k(lambda) d lambda) / integral(RSR_k(lambda) d lambda)`

where:

- `r(lambda)` = library reflectance spectrum
- `RSR_k(lambda)` = relative spectral response of Sentinel-2 band `k`

The numerical implementation may use interpolation and discrete integration.

---

# 11. Endmember Preparation

Endmembers should be:

- normalized consistently,
- checked for missing wavelength ranges,
- resampled onto sensor bands,
- labeled,
- versioned.

The FCLS matrix is:

`A in R^(C x M)`

where:

- `C` = number of Sentinel spectral bands,
- `M` = number of selected endmembers.

The observed pixel is:

`x in R^C`

The abundance vector is:

`s in R^M`.

---

# 12. Spatial Representation Data

A spatial branch may use:

- selected Sentinel bands,
- a pseudo-RGB composite,
- PCA-derived features,
- luminance/structure features.

PCA is allowed as a representation tool.

However:

> PC1 must never be treated as a mathematically guaranteed representation of all spatial detail.

The implementation should test the chosen representation empirically.

---

# 13. Spectral-Spatial Fusion Data

For each spectral band:

`B_k = bicubic(I_k)`

For the spatial branch:

`S_base = bicubic(S_input)`

`S_sr = pretrained_SR(S_input)`

`D = S_sr - S_base`

Then:

`HR_k = B_k + alpha_k * D`

The implementation must control:

- dynamic range,
- clipping,
- alpha limits,
- nodata,
- edge behavior.

---

# 14. Consistency Data

After creating `HR`, simulate the observation process:

`HR -> degradation D(.) -> Sentinel-scale prediction`

Compare:

`D(HR)` versus `I_original`

Store:

- per-band residual,
- mean absolute error,
- RMSE,
- SAM,
- maximum deviation,
- mask-aware statistics.

This residual also contributes to uncertainty estimation.

---

# 15. Uncertainty / Confidence Data

The uncertainty layer should combine interpretable signals rather than pretending to be calibrated probability unless it has been calibrated.

Candidate signals:

1. Spectral angle deviation.
2. Downsampling reconstruction error.
3. Magnitude of injected spatial residual.
4. Local instability or edge sensitivity.
5. Reference mismatch where reference imagery exists.

A normalized score can be used:

`U = w1*U_SAM + w2*U_reconstruction + w3*U_detail`

The weights must be configurable and documented.

Call it:

- uncertainty score,
- confidence score,
- reliability indicator,

rather than a formal probability of correctness unless calibration supports that claim.

---

# 16. Output Data Products

Minimum outputs:

### A. Original cube

- Sentinel-scale multispectral image.

### B. Bicubic cube

- simple 4x spatial interpolation baseline.

### C. Enhanced cube

- proposed high-resolution multispectral product.

### D. Validation maps

- SAM
- reconstruction error
- optional ERGAS-compatible summary
- per-band errors

### E. Uncertainty map

- scalar reliability layer.

### F. Abundance maps

One layer per selected FCLS endmember.

### G. Run metadata

Include:

- source scene identifier,
- bands,
- scale,
- SR model,
- model weights/version,
- preprocessing parameters,
- fusion parameters,
- consistency parameters,
- endmember set,
- validation settings.

---

# 17. File/Folder Convention

Example:

```text
data/
  raw/
    sentinel/
    reference/
    spectral_library/
    sensor_response/

  interim/
    aligned/
    masked/
    cubes/

  outputs/
    bicubic/
    sr/
    fused/
    validation/
    uncertainty/
    abundance/
    reports/
```

---

# 18. Data Integrity Checks

Every ingestion step should validate:

- expected band count,
- finite numerical values,
- plausible reflectance range,
- consistent shape after alignment,
- CRS existence,
- transform existence,
- non-empty valid-pixel mask.

Fail loudly when:

- bands are missing,
- geometry is inconsistent,
- arrays contain unexpected NaNs,
- CRS metadata is absent,
- dimensions do not match.

---

# 19. Minimal Data Needed for a 3–4 Day Prototype

A practical minimum:

1. One or two representative Sentinel-2 scenes.
2. Small AOIs rather than full scenes.
3. One usable higher-resolution reference or synthetic benchmark.
4. A small curated set of spectral endmembers.
5. Sentinel-2 sensor-response metadata.
6. One pretrained SR model.

Do not make access to a massive global dataset a prerequisite for the first working demo.

---

# 20. Data Governance

Every external dataset used in the final demo should have:

- source name,
- access method,
- licensing/use restriction,
- acquisition date,
- preprocessing steps,
- citation/reference.

Do not bundle data into the software repository if its license does not permit redistribution.
