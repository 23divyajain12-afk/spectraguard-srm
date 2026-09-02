# INTERFACES.md — SpectraGuard-SRM

Contract document for parallel development across 4 people / 4 local coding agents pushing into one shared repo. Read this before writing any module code. If you need to change a signature or schema below, that's a PR against `src/core/`, tagged for review by all 4 owners — never a silent local change.

---

## 0. Why this file exists

Four agents will be writing code at the same time against modules that don't exist yet. The only way that works without constant merge conflicts and integration breakage is:

1. **Everyone agrees on the data shapes crossing module boundaries** (Section 2) before writing logic.
2. **Everyone owns a disjoint set of folders** (Section 4) — you only ever edit files inside your own folder, plus tests for your own folder.
3. **A tiny synthetic fixture exists on Day 1 morning** (Section 6) so nobody blocks on anybody else's real implementation to start writing and testing their own module.
4. **Shared files (`src/core/`, `src/pipeline/orchestrator.py`) are edited rarely, in small PRs, with review from whoever else touches them** (Section 5).

If your function matches the signature and returns the right dataclass, your module is done from everyone else's point of view — regardless of what's inside it.

---

## 1. Repo layout

```text
spectraguard-srm/
├── README.md
├── INTERFACES.md
├── requirements.txt
├── config/
│   └── default.yaml
├── data/
│   ├── raw/{sentinel,reference,spectral_library,sensor_response}/
│   ├── interim/{aligned,masked,cubes}/
│   └── outputs/{bicubic,sr,fused,validation,uncertainty,abundance,reports}/
├── src/
│   ├── core/            # schemas.py, config.py, logging_utils.py
│   ├── data/             # sentinel_io.py, preprocessing.py, geospatial.py
│   ├── sr/                # model_adapter.py, tiling.py, spatial_representation.py
│   ├── fusion/           # bicubic.py, detail_residual.py, spectral_injection.py
│   ├── consistency/      # degradation.py, projection.py
│   ├── validation/       # metrics.py, uncertainty.py, spectral_metrics.py
│   ├── library/           # usgs_io.py, sensor_response.py, endmember_preparation.py
│   ├── unmixing/          # fcls.py
│   ├── app/                # dashboard.py
│   └── pipeline/           # orchestrator.py
├── tests/
│   ├── fixtures/           # synthetic small-AOI cube + toy endmembers (see §6)
│   ├── data/ sr/ fusion/ consistency/ validation/ library/ unmixing/
└── notebooks/               # scratch/demo, not depended on by src/
```

Create all folders above (with empty `.gitkeep` where needed) in the first commit to `main`, before anyone branches off.

---

## 2. Shared data contracts (`src/core/schemas.py`)

Every function that crosses a module boundary takes and returns one of these — never a bare `np.ndarray` for anything that used to be a raster, and never a bare dict for anything below. This is what lets four people build against each other's *interfaces* without each other's *code*.

```python
# src/core/schemas.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Protocol
import numpy as np

@dataclass
class SentinelCube:
    """The one raster type that flows through the whole pipeline. Every
    raster-producing function must preserve or explicitly update every field —
    never silently drop CRS/transform/mask. (Data.md §6, Skills.md Rule 2)"""
    data: np.ndarray                        # (C, H, W) float32 reflectance
    band_names: List[str]                   # order matches axis 0 of `data`
    crs: str                                # e.g. "EPSG:32643"
    transform: Tuple[float, float, float, float, float, float]  # affine 6-tuple
    resolution_m: float                     # working GSD for this cube
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    mask: np.ndarray                        # (H, W) bool, True = valid pixel
    nodata: Optional[float] = None
    acquisition_time: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # scene_id, resample_method, cloud_pct...

@dataclass
class FusedCube(SentinelCube):
    """Output of spectral-spatial fusion / consistency projection."""
    alpha_map: Optional[np.ndarray] = None   # (C,H,W) or (H,W) injection coeffs used
    provenance: str = "bicubic+residual"
    anomaly_mask: Optional[np.ndarray] = None

@dataclass
class SpatialRepresentation:
    array: np.ndarray            # (H,W) or (3,H,W), normalized for the SR model
    method: str                  # "pseudo_rgb" | "pca" | "single_band"
    source_bands: List[str]
    normalization: Dict[str, Any]  # min/max or mean/std, needed to invert later

@dataclass
class SRResidual:
    D: np.ndarray                # high-frequency residual (SR - bicubic)
    sr_output: np.ndarray
    bicubic_base: np.ndarray
    model_name: str
    model_version: str
    scale: int

@dataclass
class DegradationPrediction:
    predicted: SentinelCube      # D(HR) simulated back at Sentinel scale
    degradation_method: str
    kernel_params: Dict[str, Any]

@dataclass
class ValidationMetrics:
    sam_mean: float
    sam_map: np.ndarray
    per_band_rmse: Dict[str, float]
    per_band_mae: Dict[str, float]
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    ergas: Optional[float] = None
    # never "ground_truth" unless it actually is
    reference_type: str = "internal_consistency"  # | "reference_imagery" | "synthetic_benchmark"

@dataclass
class UncertaintyMap:
    U: np.ndarray                        # (H,W) normalized [0,1]
    components: Dict[str, np.ndarray]    # {"sam":..., "reconstruction":..., "detail":...}
    weights: Dict[str, float]
    label: str = "uncertainty_score"     # not "probability" unless calibrated

@dataclass
class RawSpectrum:
    name: str
    material_class: str
    wavelength_nm: np.ndarray
    reflectance: np.ndarray
    source_meta: Dict[str, Any]

@dataclass
class SensorResponse:
    band_names: List[str]
    wavelength_nm: np.ndarray
    rsr: Dict[str, np.ndarray]           # band_name -> response curve on wavelength_nm grid

@dataclass
class EndmemberSet:
    A: np.ndarray                        # (C, M) Sentinel-band-integrated matrix
    names: List[str]                     # length M
    material_classes: List[str]
    source_spectra: List[str]

@dataclass
class AbundanceResult:
    S: np.ndarray                        # (M, H, W); sum over M == 1 per valid pixel
    endmember_names: List[str]
    residual: Optional[np.ndarray] = None
    solver: str = "fcls"                 # must reflect the actual constraints used

@dataclass
class RunMetadata:
    scene_id: str
    aoi: Dict[str, Any]
    bands: List[str]
    model_name: str
    model_version: str
    scale: int
    fusion_params: Dict[str, Any]
    consistency_params: Dict[str, Any]
    endmember_set: str
    validation_settings: Dict[str, Any]
    timestamp: str
    software_versions: Dict[str, str]

class SRModel(Protocol):
    """Anything satisfying this is swappable — Real-ESRGAN/RCAN/EDSR/SwinIR/etc."""
    def predict(self, spatial_input: np.ndarray) -> np.ndarray: ...
```

`src/core/config.py` mirrors the Phase-0 YAML (`RunConfig` dataclass with `project`, `data`, `sr`, `fusion`, `consistency`, `unmixing` sections) — same shape as the config in the architecture doc. Freeze this by end of Day 0.

---

## 3. Module interfaces by folder

Each row is a function contract. Build against these signatures with the fixture from §6 before the upstream module is real — a stub that returns a correctly-shaped fixture object is a legitimate Day-1 deliverable.

### `src/data/` — Owner: Person A

| File | Function | Signature |
|---|---|---|
| `sentinel_io.py` | `load_sentinel` | `(scene_path: str, config: RunConfig) -> SentinelCube` |
| `preprocessing.py` | `preprocess` | `(cube: SentinelCube, config: RunConfig) -> SentinelCube` |
| `geospatial.py` | `reproject_match` | `(src: SentinelCube, target_crs: str, target_transform: tuple) -> SentinelCube` |
| `geospatial.py` | `crop_to_aoi` | `(cube: SentinelCube, aoi_geom) -> SentinelCube` |

Contract: `preprocess()` output has every band on the same grid at `config.data.working_resolution_m`, mask merged with cloud/shadow flags, reflectance as float `[0,1]`.

### `src/sr/` + `src/fusion/` — Owner: Person B

| File | Function | Signature |
|---|---|---|
| `fusion/bicubic.py` | `bicubic_upscale` | `(cube: SentinelCube, scale: int) -> SentinelCube` |
| `sr/spatial_representation.py` | `build_spatial_input` | `(cube: SentinelCube, config: RunConfig) -> SpatialRepresentation` |
| `sr/model_adapter.py` | `load_sr_model` | `(config: RunConfig) -> SRModel` |
| `sr/tiling.py` | `tiled_inference` | `(model: SRModel, image: np.ndarray, tile_size: int, overlap: int) -> np.ndarray` |
| `fusion/detail_residual.py` | `extract_residual` | `(sr_output: np.ndarray, bicubic_base: np.ndarray) -> SRResidual` |
| `fusion/spectral_injection.py` | `fuse_multispectral` | `(bicubic_cube: SentinelCube, spatial_base: np.ndarray, spatial_hr: np.ndarray, config: RunConfig) -> FusedCube` |
| `fusion/spectral_injection.py` | `run_safety_checks` | `(fused: FusedCube, config: RunConfig) -> FusedCube` (sets `anomaly_mask`, clips reflectance) |

Contract: `SRModel.predict()` always takes a normalized `(H,W)` or `(3,H,W)` array and returns the same channel count at `config.sr.scale`× resolution — this is what keeps the backbone swappable per Skills.md Rule 3.

### `src/consistency/` + `src/validation/` — Owner: Person C

| File | Function | Signature |
|---|---|---|
| `consistency/degradation.py` | `degrade` | `(hr_cube: FusedCube, config: RunConfig) -> DegradationPrediction` |
| `consistency/projection.py` | `project_to_measurement_consistency` | `(hr_cube: FusedCube, original_cube: SentinelCube, config: RunConfig) -> FusedCube` |
| `validation/spectral_metrics.py` | `compute_sam` | `(x: np.ndarray, y: np.ndarray) -> np.ndarray` |
| `validation/spectral_metrics.py` | `compute_band_errors` | `(pred: SentinelCube, ref: SentinelCube) -> Dict[str, float]` |
| `validation/metrics.py` | `evaluate` | `(hr_cube: FusedCube, original_cube: SentinelCube, reference: Optional[SentinelCube], config: RunConfig) -> ValidationMetrics` |
| `validation/uncertainty.py` | `estimate_uncertainty` | `(hr_cube: FusedCube, metrics: ValidationMetrics, config: RunConfig) -> UncertaintyMap` |

Contract: `ValidationMetrics.reference_type` must honestly reflect what was actually compared — `"internal_consistency"` unless real reference/synthetic-benchmark data was used (Data.md §9, Skills.md Rule 1).

### `src/library/` + `src/unmixing/` + `src/app/` — Owner: Person D

| File | Function | Signature |
|---|---|---|
| `library/usgs_io.py` | `load_usgs_spectrum` | `(path: str) -> RawSpectrum` |
| `library/sensor_response.py` | `load_sensor_response` | `(path: str, band_names: List[str]) -> SensorResponse` |
| `library/endmember_preparation.py` | `prepare_endmembers` | `(library: List[RawSpectrum], sensor_response: SensorResponse, config: RunConfig) -> EndmemberSet` |
| `unmixing/fcls.py` | `fcls` | `(hr_cube: FusedCube, endmembers: EndmemberSet, config: RunConfig) -> AbundanceResult` |
| `app/dashboard.py` | *(none exported)* | reads saved artifacts from `data/outputs/` only — a leaf consumer, not a dependency of anything |

Contract: `fcls()` must enforce **both** `s_i ≥ 0` **and** `Σ s_i = 1`. A bare `scipy.optimize.nnls` call satisfies neither the sum-to-one constraint nor the `solver="fcls"` label (Skills.md §6, Rule 1).

### `src/pipeline/orchestrator.py` — Owner: Person A (integration lead)

```python
def run_pipeline(config: RunConfig) -> RunMetadata: ...
```

A straight-line call through every function above, phase by phase. This file is intentionally thin — it should almost never need logic changes, only new call sites as modules land. See §5 for how to touch it safely.

---

## 4. Folder ownership & agent assignment

Each person runs their own local coding agent scoped to their row — the agent's job is exactly that file list. Nobody's agent edits another owner's folder; if you find a bug in someone else's module, open an issue/PR against their branch instead of pushing a fix yourself.

| Person | Role | Owns (folders) | Architecture phases | Day focus |
|---|---|---|---|---|
| **A** | Data + Integration Lead | `src/core/`\*, `src/data/`, `src/pipeline/`\*, `config/`, `tests/fixtures/` | P0 Setup, P1 Ingestion, P2 Geospatial preprocessing | Day 1 AM: ship the fixture + loader first — everyone else is blocked on this |
| **B** | SR + Fusion Lead | `src/sr/`, `src/fusion/` | P3 Bicubic, P4 Spatial rep, P5 SR inference, P6 Residual, P7 Injection, P8 Safety checks | Day 1–2: bicubic + SR model running, then the `HR_k = B_k + α_k·D` fusion |
| **C** | Consistency + Validation Lead | `src/consistency/`, `src/validation/` | P9 Degradation, P10 Projection, P11 Spectral validation, P12 Reference validation, P13 Uncertainty | Day 2: degrade→compare→correct loop, then SAM/RMSE/uncertainty |
| **D** | Library + Unmixing + App Lead | `src/library/`, `src/unmixing/`, `src/app/` | P14 Endmembers, P15 FCLS, P16–17 Dashboard, P18 Packaging | Day 1–2: build the dashboard skeleton against fixture outputs so it's not a Day-3 bottleneck; Day 3: wire in real FCLS + abundance maps |

\* `src/core/` and `src/pipeline/orchestrator.py` are shared boundary files — see §5 before editing them.

---

## 5. Git workflow

- **Branches:** `feat/data-ingestion` (A), `feat/sr-fusion` (B), `feat/consistency-validation` (C), `feat/unmixing-app` (D), all off `dev`. `main` is protected/release-only.
- **Shared files (`src/core/*`, `src/pipeline/orchestrator.py`, `config/default.yaml`):** small, isolated commits only; open a PR and tag the other 3 owners before merging. Never bundle a schema change into a feature commit.
- **Never edit inside another owner's folder** — send them a PR comment or a failing test instead.
- **Integration checkpoints:** merge all 4 branches into `dev` at the end of each day (matches the Day 1–4 plan in Memory.md/System_Architecture.md §24–27), run `orchestrator.run_pipeline()` against the fixture as a smoke test, fix breakage together before starting the next day's branch work.
- **PR requirement:** a module's PR into `dev` must pass its own `tests/<module>/` against the fixture in §6 before merge — not against real Sentinel data, which may not exist yet.

---

## 6. Fixture-first strategy (unblocks Day 1)

Person A commits `tests/fixtures/aoi_small.npz` (or `.pkl`) by **Day 1 late morning**: a tiny synthetic `SentinelCube` — e.g. 4–6 bands, 64×64 px, plausible dummy CRS/transform/mask, values in `[0,1]`. Also drop a toy `EndmemberSet` (2–3 synthetic spectra) in the same folder.

Every other module writes and unit-tests its function against this fixture immediately, without waiting for the real loader, real SR weights, or real spectral library to be ready. Swapping the fixture for real data at integration time should require zero code changes downstream — that's the whole point of §2.

---

## 7. Definition of done (per module)

Matches Skills.md — a module isn't done until:

- it runs on the Day-1 fixture end-to-end,
- its output matches the dataclass contract in §2 exactly (right fields populated, right shapes),
- outputs are saved with metadata under `data/outputs/`,
- failures fail loudly (no silent NaNs, no silently dropped CRS),
- assumptions (resampling method, degradation kernel, alpha formula choice, etc.) are written down in code comments or `RunMetadata`,
- and it's reproducible from `config/default.yaml` alone.
