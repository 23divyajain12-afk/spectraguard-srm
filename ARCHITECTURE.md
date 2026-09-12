# SpectraGuard-SRM Architecture

This document describes the production architecture of the SpectraGuard-SRM workspace as implemented in the current repository. It reflects the actual code structure, execution flow, and runtime configuration used by the project.

---

## 1. Repository Layout and Component Mapping

### 1.1 Root tree and major components

```text
spectraguard-srm/
├── README.md                         # Quick start, execution commands, and operational notes
├── Data.md                          # Scientific data model and Sentinel-2 processing assumptions
├── INTERFACES.md                    # Interface contracts and cross-module responsibilities
├── Memory.md                        # Project memory, design rationale, and constraints
├── Skills.md                        # Project goals and scientific/engineering requirements
├── System_Architecture.md           # Higher-level design notes and conceptual architecture
├── Spectra-Guard_UI_Reference.md     # UI/UX constraints for the dashboard and processing AOI
├── config/
│   ├── default.yaml                 # Primary runtime configuration for datasets, SR, fusion, consistency, and unmixing
│   └── s2c_endmembers.yaml          # Optional endmember manifest / material metadata for spectral workflows
├── data/
│   ├── raw/
│   │   ├── S2C_MSIL2A_20260831T052641_N0512_R105_T43QCA_20260831T102215.SAFE/
│   │   │   ├── GRANULE/             # Sentinel-2 tile imagery and metadata
│   │   │   ├── DATASTRIP/           # Strip-level metadata and ancillary files
│   │   │   ├── HTML/                # Rendered product metadata and human-readable docs
│   │   │   ├── INSPIRE.xml          # Geospatial metadata
│   │   │   ├── manifest.safe        # SAFE manifest file
│   │   │   └── MTD_MSIL2A.xml       # Main Sentinel-2 metadata file
│   │   ├── sensor_response/         # Sentinel-2 spectral response files
│   │   └── spectral_library/        # Endmember sources and manifest metadata
│   └── outputs/
│       ├── abundance/               # FCLS abundance outputs
│       ├── bicubic/                 # Bicubic baseline interpolation results
│       ├── fused/                   # Spectrally fused and consistency-adjusted cube
│       ├── reports/                 # Run metadata about the current execution
│       ├── sr/                      # Super-resolution model outputs per tile/array
│       ├── uncertainty/             # Uncertainty maps and component diagnostics
│       ├── validation/              # Original cube, metrics, and validation artifacts
│       ├── comparison.png           # Generated comparison visualization
│       ├── comparison_stretched.png # Stretched spectral comparison visualization
│       ├── ground_detail_comparison.png
│       ├── ground_detail_comparison_nir.png
│       └── ground_detail_true_color.png
├── notebooks/                       # Exploration notebooks (not required for main pipeline execution)
├── src/
│   ├── app/
│   │   ├── dashboard.py             # Artifact inspection utilities for the dashboard layer
│   │   ├── ui_readme.md             # README for the lightweight static UI
│   │   └── web_ui.py                # Read-only web interface for scene/readouts and artifact previewing
│   ├── consistency/
│   │   ├── degradation.py           # Simulation of the observed Sentinel-2 measurement from HR output
│   │   └── projection.py            # Consistency projection to recover measurement fidelity after fusion
│   ├── core/
│   │   ├── config.py                # YAML-to-dataclass configuration loader
│   │   └── schemas.py               # Strongly typed dataclasses for cubes, metrics, and metadata
│   ├── data/
│   │   ├── geospatial.py            # AOI cropping, reprojection, and geometry alignment
│   │   ├── preprocessing.py         # Band alignment and working-grid resampling
│   │   ├── sentinel_io.py           # Sentinel SAFE ingestion for B02/B03/B04/B08/B11 loading
│   │   └── __pycache__/             # Python bytecode cache
│   ├── fusion/
│   │   ├── bicubic.py               # Bicubic baseline upsampling for comparison and initialization
│   │   ├── detail_residual.py       # Difference extraction between SR output and bicubic base
│   │   └── spectral_injection.py    # Detail injection and anomaly/safety checks
│   ├── library/
│   │   ├── endmember_preparation.py # Endmember preparation and spectral band selection
│   │   ├── s2c_usgs_library.py      # Sentinel-2C-aware endmember loading and preparation logic
│   │   ├── sensor_response.py       # Spectral response function loading and band resampling
│   │   └── usgs_io.py               # USGS spectrum ingestion and parsing
│   ├── pipeline/
│   │   └── orchestrator.py          # End-to-end pipeline coordinator and artifact writer
│   ├── sr/
│   │   ├── model_adapter.py         # Model backend adapter (Sen2SR or interpolation fallback)
│   │   ├── spatial_representation.py # Spatial input normalization and pseudo-RGB / band shaping
│   │   ├── tiling.py                # Overlapping-tile SR inference and overlap averaging
│   │   └── __pycache__/             # Python bytecode cache
│   ├── unmixing/
│   │   └── fcls.py                  # Fully constrained least squares spectral unmixing
│   ├── validation/
│   │   ├── metrics.py               # Validation metrics aggregation (PSNR, SSIM, ERGAS, SAM)
│   │   ├── spectral_metrics.py      # Core SAM and band error computations
│   │   ├── uncertainty.py           # Uncertainty weighting and map synthesis
│   │   └── __pycache__/             # Python bytecode cache
│   └── __pycache__/                 # Python bytecode cache
├── tests/
│   ├── adversarial/                 # Edge-case and failure-mode tests
│   ├── app/                         # UI/read-only dashboard tests
│   ├── consistency/                 # Consistency projection checks
│   ├── data/                        # Ingestion, AOI, and SAFE-window tests
│   ├── fixtures/                    # Synthetic cube fixture loader and small-AOI data
│   ├── fusion/                      # Fusion and detail-injection tests
│   ├── library/                     # Endmember and sensor-response tests
│   ├── pipeline/                    # Orchestrator-level execution tests
│   ├── sr/                          # Super-resolution and tiling tests
│   ├── unmixing/                    # FCLS and adversarial unmixing tests
│   └── validation/                  # Validation metric and external-reference tests
├── .gitignore                       # Git ignore rules
├── pytest.ini                       # Pytest configuration
├── opensr-ldsrs2_v1_0_0.ckpt        # Optional SEN2SR checkpoint asset
├── requirements.txt                 # Core dependency set
├── requirements-sen2sr.txt          # Optional SEN2SR-specific dependencies
└── ARCHITECTURE.md                  # This document
```

### 1.2 Key file responsibility summary

| File | Responsibility |
| --- | --- |
| [config/default.yaml](config/default.yaml) | Defines all runtime behavior: bands, AOI, model, scale, tile size, overlap, device, fusion weights, and unmixing inputs. |
| [src/core/config.py](src/core/config.py) | Loads YAML into strongly typed configuration objects used throughout the pipeline. |
| [src/core/schemas.py](src/core/schemas.py) | Defines the shared data contracts: SentinelCube, FusedCube, ValidationMetrics, RunMetadata, and endmember structures. |
| [src/data/sentinel_io.py](src/data/sentinel_io.py) | Reads Sentinel-2 SAFE products for B02, B03, B04, B08, and B11, rescales reflectance, and preserves geographic metadata. |
| [src/data/geospatial.py](src/data/geospatial.py) | Reprojects and crops the cube to the configured AOI or target transform, keeping spatial alignment valid. |
| [src/data/preprocessing.py](src/data/preprocessing.py) | Resamples all bands to the configured working grid and merges cloud/shadow/quality masks. |
| [src/sr/model_adapter.py](src/sr/model_adapter.py) | Wraps the SR backend: Sen2SR when configured, otherwise a lightweight interpolation fallback. |
| [src/sr/spatial_representation.py](src/sr/spatial_representation.py) | Normalizes and prepares the SR model input as a pseudo-RGB or single-band spatial array. |
| [src/sr/tiling.py](src/sr/tiling.py) | Runs model inference tile-by-tile with overlap, then averages overlap regions to maintain stable memory usage. |
| [src/fusion/bicubic.py](src/fusion/bicubic.py) | Produces the lower-resolution baseline that anchors the super-resolution estimate. |
| [src/fusion/detail_residual.py](src/fusion/detail_residual.py) | Extracts the high-frequency detail residual between the SR estimate and the bicubic baseline. |
| [src/fusion/spectral_injection.py](src/fusion/spectral_injection.py) | Injects learned spatial detail into the multispectral cube using band-wise alpha coefficients and validity checks. |
| [src/consistency/projection.py](src/consistency/projection.py) | Projects the fused result back toward the Sentinel observation using a consistency loop. |
| [src/validation/metrics.py](src/validation/metrics.py) | Aggregates SAM, PSNR, SSIM, ERGAS, and band-wise error metrics for validation. |
| [src/validation/spectral_metrics.py](src/validation/spectral_metrics.py) | Implements the low-level SAM and spectral error logic used by the validation layer. |
| [src/validation/uncertainty.py](src/validation/uncertainty.py) | Estimates uncertainty from reconstruction residuals, SAM, and detail magnitude. |
| [src/unmixing/fcls.py](src/unmixing/fcls.py) | Computes abundance maps with Fully Constrained Least Squares (FCLS) for endmember interpretation. |
| [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) | Owns the end-to-end workflow and writes all artifacts to data/outputs. |
| [src/app/web_ui.py](src/app/web_ui.py) | Serves a read-only dashboard that visualizes AOI, scene metadata, and processed outputs without altering scientific data. |
| [tests/](tests/) | Contains regression, pipeline, fusion, validation, and edge-case checks to verify correctness and stability. |

> Note: there is no dedicated [plot_comparison.py](README.md) file in the current repository. The project produces the comparison plots directly under [data/outputs](data/outputs), and the UI in [src/app/web_ui.py](src/app/web_ui.py) exposes the artifacts for inspection.

---

## 2. End-to-End Pipeline Workflow

### 2.1 Pipeline architecture diagram

```mermaid
flowchart TD
    A[Sentinel-2 L2A .SAFE product] --> B[load_sentinel() in src/data/sentinel_io.py]
    B --> C[AOI crop + bounding-box validation in src/data/geospatial.py]
    C --> D[Resample + working-grid preprocessing in src/data/preprocessing.py]
    D --> E[Build spatial SR input via src/sr/spatial_representation.py]
    E --> F[Optional SR model adapter in src/sr/model_adapter.py]
    F --> G[Overlapping tiled inference in src/sr/tiling.py]
    G --> H[Baseline bicubic interpolation in src/fusion/bicubic.py]
    H --> I[Residual detail extraction in src/fusion/detail_residual.py]
    E --> J[Detail injection in src/fusion/spectral_injection.py]
    H --> J
    J --> K[Safety clipping and anomaly masking]
    K --> L[Measurement consistency projection in src/consistency/projection.py]
    L --> M[Validation: SAM, RMSE, PSNR, SSIM, ERGAS in src/validation/metrics.py]
    M --> N[Uncertainty estimation in src/validation/uncertainty.py]
    L --> O[Optional FCLS abundance estimation in src/unmixing/fcls.py]
    M --> P[Save artifacts under data/outputs/]
    N --> P
    O --> P
    P --> Q[Visualization and dashboard: data/outputs plots + src/app/web_ui.py]
    Q --> R[Scientific review and downstream analysis]
```

### 2.2 Stage-by-stage technical walkthrough

#### Stage 1 — Data ingestion and AOI bounding-box cropping

The project loads Sentinel-2 imagery from either a SAFE directory or a prebuilt NPZ fixture. The primary code is in [src/data/sentinel_io.py](src/data/sentinel_io.py), and the AOI logic sits in [src/data/geospatial.py](src/data/geospatial.py).

Key behavior:

- The SAFE loader locates B02, B03, B04, B08, and B11 products.
- It reads the metadata, acquisition time, and band-specific transforms.
- It converts DN values into reflectance by dividing by 10,000.
- It applies a valid-pixel mask and recognizes invalid/no-data pixels.
- It clips the raster to the configured AOI bounds if `data.aoi` is set in [config/default.yaml](config/default.yaml).
- It validates that the AOI intersects the scene and rejects empty or invalid boxes.

This stage creates a `SentinelCube` object that carries:

- `data`: array shaped as `(bands, height, width)`
- `band_names`: ordered band identifiers such as `B02`, `B03`, `B04`, `B08`, `B11`
- `crs`, `transform`, `bounds`, `mask`, `nodata`, `meta`

The pipeline then calls the geometry alignment and crop functions before moving into preprocessing.

#### Stage 2 — Preprocessing and working-grid normalization

The preprocessing stage in [src/data/preprocessing.py](src/data/preprocessing.py) converts input imagery to the working grid defined by `data.working_resolution_m`.

Important implementation details:

- It rescales each band using interpolation to match the configured working resolution.
- It combines source masks with cloud/shadow/quality masks.
- It clips values into `[0, 1]` reflectance range to keep the model and fusion components numerically stable.
- It updates metadata such as `working_resolution_m` and `resample_method`.

This ensures the spatial branch and spectral fusion operate on a homogeneous, valid, shared grid.

#### Stage 3 — Super-resolution spatial representation

The super-resolution path begins with [src/sr/spatial_representation.py](src/sr/spatial_representation.py), which normalizes the cube into a model-ready representation.

The implementation:

- selects an SR-relevant input set (`B02/B03/B04/B08` for SEN2SR, or the configured data bands otherwise)
- masks invalid pixels before normalization
- computes per-channel min/max and rescales values to `[0, 1]`
- returns a `SpatialRepresentation` object with `array`, `source_bands`, and normalization metadata

This is the input to the super-resolution backbone.

#### Stage 4 — Overlapping tiled super-resolution inference

The actual SR inference occurs through [src/sr/tiling.py](src/sr/tiling.py) and the adapter in [src/sr/model_adapter.py](src/sr/model_adapter.py).

The model adapter supports:

- `sen2sr` backend using the official OpenSR latent diffusion stack
- a lightweight interpolation fallback when no external model is configured

The tile-based inference is central to stability:

- it processes a large image in overlapping tiles
- uses `tile_size - overlap` as the step size
- computes predictions tile by tile
- accumulates predicted values into an output array
- averages overlap regions with a weight mask so seam artifacts are reduced
- makes progress callback reporting possible for long jobs

This approach is a direct VRAM-stability strategy: instead of passing the full high-resolution image to the model in one pass, the system keeps the working set bounded by the configured tile size and overlap, preventing GPU memory spikes on cards like the RTX 3050.

#### Stage 5 — Bicubic baseline and high-frequency residual injection

The pipeline creates a bicubic baseline via [src/fusion/bicubic.py](src/fusion/bicubic.py) and computes a residual from the SR output in [src/fusion/detail_residual.py](src/fusion/detail_residual.py).

The logic is:

- compute a lower-quality but spatially coherent baseline image
- estimate the high-frequency detail as `SR - bicubic`
- use this residual as the source of sharpened detail to inject into the original multispectral and/or band subset

This stage is designed to preserve the original observation’s spectral structure while adding model-derived spatial detail.

#### Stage 6 — Spectral injection and safety checks

The detail injection logic is implemented in [src/fusion/spectral_injection.py](src/fusion/spectral_injection.py).

Critical design decisions:

- The residual is injected with band-wise alpha coefficients.
- The alpha values are constrained by `alpha_min` and `alpha_max` from [config/default.yaml](config/default.yaml).
- A small `epsilon` prevents division-by-zero during covariance normalization.
- The fused cube is clipped to `[0, 1]` reflectance range.
- Invalid or non-finite values are marked in an anomaly mask.

This keeps the HR result bounded and science-aware rather than letting purely generated detail distort reflectance values.

#### Stage 7 — Measurement consistency projection

Once fused, the output is refined in [src/consistency/projection.py](src/consistency/projection.py).

This loop:

- models the measurement-consistency degradation of the HR cube
- compares it to the original Sentinel observation
- computes a residual error map
- redistributes the correction using interpolation and mask filtering
- applies a variable correction factor controlled by `consistency.lambda_`

This makes the final enhanced cube remain accountable to the measured observation rather than behaving as a purely unconstrained generative image.

#### Stage 8 — Spatial and spectral quality evaluation

Validation is performed by [src/validation/metrics.py](src/validation/metrics.py) and low-level metric helpers in [src/validation/spectral_metrics.py](src/validation/spectral_metrics.py).

The code computes:

- SAM (Spectral Angle Mapper): measures spectral similarity between vectors
- RMSE and MAE per band: quantifies band-wise reconstruction error
- PSNR: measures pixel-level fidelity in a logarithmic scale
- SSIM: measures structural similarity over the valid mask
- ERGAS: global relative dimensionless synthesis error

These metrics are stored in:

- [data/outputs/validation/metrics.json](data/outputs/validation/metrics.json)
- [data/outputs/validation/metrics.npz](data/outputs/validation/metrics.npz)

This stage is the quantitative proof layer of the pipeline.

#### Stage 9 — Uncertainty and unmixing

After validation:

- [src/validation/uncertainty.py](src/validation/uncertainty.py) estimates uncertainty using reconstruction, SAM, and detail magnitude contributions.
- [src/unmixing/fcls.py](src/unmixing/fcls.py) optionally runs FCLS endmember-based abundance estimation when configured.

When endmember definitions are present, output abundance maps are written to [data/outputs/abundance](data/outputs/abundance).

#### Stage 10 — Artifact export and visualization

The orchestrator writes the final artifacts to [data/outputs](data/outputs):

- original cube
- bicubic baseline
- fused cube
- SR prediction
- validation metrics
- uncertainty values
- abundance maps
- run metadata report

The project also exports generated plot PNGs into [data/outputs](data/outputs) for visual comparison. The user-facing viewer is [src/app/web_ui.py](src/app/web_ui.py), which is a read-only inspection dashboard and not a scientific editing tool.

---

## 3. Technical Design Decisions

### 3.1 Tiling and memory management strategy

The memory-safe processing strategy is centered on [src/sr/tiling.py](src/sr/tiling.py):

1. The image is split into overlapping tiles according to `tile_size` and `overlap`.
2. The model runs on one tile at a time, keeping the receptive tensor bounded.
3. Each tile prediction is accumulated into a larger output array.
4. Overlap regions are averaged using a weight map instead of simply overwriting.
5. This avoids GPU memory blow-up and keeps the runtime practical on mid-range hardware such as the RTX 3050.

This design is especially important because the project uses latent-diffusion style SR models and high-resolution multispectral arrays that would otherwise exceed typical VRAM budgets if processed as a single giant image.

### 3.2 Role of configuration in driving execution

The entire runtime is driven by [config/default.yaml](config/default.yaml), which is converted into runtime dataclasses by [src/core/config.py](src/core/config.py).

The configuration controls:

- `data.bands`: spectral subset and target grid
- `data.scene_path`: location of the Sentinel SAFE product or fixture
- `data.aoi`: fixed AOI bounds used for cropping
- `sr.model`: super-resolution backend (`sen2sr` or fallback)
- `sr.scale`: upsampling factor, with SEN2SR requiring `4`
- `sr.tile_size` and `sr.overlap`: VRAM-safe processing windowing
- `sr.device`: `auto`, `cpu`, or `cuda`
- `sr.checkpoint`: pretrained model checkpoint path or identifier
- `fusion.alpha_min` and `fusion.alpha_max`: bounds on detail injection amplitude
- `consistency.iterations` and `consistency.lambda_`: control how strongly the fused cube is projected back toward the measured observation
- `unmixing.endmembers` and `unmixing.sensor_response`: sensor-aware unmixing setup

The orchestrator in [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) reads the config and executes the same pipeline across fixture and real-world Sentinel scenes without rewriting code paths.

---

## 4. Execution Model and Data Contract

The codebase uses a small set of robust data contracts to keep operations consistent:

- `SentinelCube`: source or processed observation tensor with metadata, transform, and mask
- `FusedCube`: enhanced HR data after spectral injection and consistency projection
- `ValidationMetrics`: quantitative summary from the metric evaluation stage
- `RunMetadata`: serializable run summary exported as JSON for reproducibility

This shared contract enables the orchestrator to safely move data between ingestion, SR, fusion, validation, uncertainty, and abundance modules without hidden assumptions.

---

## 5. Architectural Summary

The architecture is intentionally split between a pretrained spatial model and a measurement-aware analytical pipeline:

- the SR model estimates the high-frequency spatial structure;
- the fusion layer injects it in a controlled, bounded way;
- the consistency projection keeps the output tied to the actual Sentinel observation;
- validation quantifies the result scientifically;
- the unmixing stage adds material interpretation when endmembers are provided;
- the UI and exported artifacts make the output inspectable and reproducible.

This approach is appropriate for a practical, low-training Sentinel-2 super-resolution and spectral-analysis pipeline that must remain stable, explainable, and deployable on moderate GPU hardware.
