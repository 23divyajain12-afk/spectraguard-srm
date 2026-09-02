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
