from typing import Optional

import numpy as np

from src.consistency.degradation import degrade
from src.core.config import RunConfig
from src.core.schemas import FusedCube, SentinelCube, ValidationMetrics
from src.validation.spectral_metrics import compute_band_errors, compute_sam


def _validate_matching_grid(pred: SentinelCube, ref: SentinelCube) -> None:
    if pred.data.shape != ref.data.shape:
        raise ValueError("prediction and reference cubes must have matching shapes")
    if pred.band_names != ref.band_names:
        raise ValueError("prediction and reference bands must match in order")
    if pred.crs != ref.crs:
        raise ValueError("prediction and reference CRS must match")
    if not np.allclose(pred.transform, ref.transform):
        raise ValueError("prediction and reference transforms must match")
    if not np.isclose(pred.resolution_m, ref.resolution_m):
        raise ValueError("prediction and reference resolutions must match")
    if not np.allclose(pred.bounds, ref.bounds):
        raise ValueError("prediction and reference bounds must match")
    if pred.mask.shape != pred.data.shape[1:] or ref.mask.shape != ref.data.shape[1:]:
        raise ValueError("prediction and reference masks must match spatial dimensions")


def _valid_mask(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask & np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(ref), axis=0)


def _global_ssim(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    values = []
    c1 = 0.01**2
    c2 = 0.03**2
    for index in range(pred.shape[0]):
        x = pred[index][mask]
        y = ref[index][mask]
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        variance_x = float(np.var(x))
        variance_y = float(np.var(y))
        covariance = float(np.mean((x - mean_x) * (y - mean_y)))
        numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
        denominator = (mean_x**2 + mean_y**2 + c1) * (
            variance_x + variance_y + c2
        )
        values.append(numerator / denominator if denominator > 0 else 1.0)
    return float(np.mean(values))


def evaluate(
    hr_cube: FusedCube,
    original_cube: SentinelCube,
    reference: Optional[SentinelCube],
    config: RunConfig,
) -> ValidationMetrics:
    """Evaluate the degraded HR result against the original observation."""
    predicted = degrade(hr_cube, config).predicted
    if predicted.band_names != original_cube.band_names:
        raise ValueError("degraded prediction and original bands must match in order")
    if predicted.data.shape != original_cube.data.shape:
        raise ValueError("degraded prediction and original cube must align")
    mask = _valid_mask(predicted.data, original_cube.data, predicted.mask & original_cube.mask)
    if not np.any(mask):
        raise ValueError("no valid pixels available for evaluation")

    sam_map = compute_sam(predicted.data, original_cube.data)
    sam_map = np.where(mask, sam_map, np.nan)
    errors = compute_band_errors(predicted, original_cube)
    mae = {
        name: float(np.mean(np.abs(predicted.data[index][mask] - original_cube.data[index][mask])))
        for index, name in enumerate(predicted.band_names)
    }
    reference_type = "internal_consistency"
    psnr = ssim = ergas = None
    if reference is not None:
        _validate_matching_grid(hr_cube, reference)
        reference_type = str(reference.meta.get("reference_type", "external_reference"))
        if reference_type not in {"external_reference", "synthetic_benchmark"}:
            raise ValueError(f"unsupported reference_type: {reference_type}")
        reference_mask = _valid_mask(
            hr_cube.data, reference.data, hr_cube.mask & reference.mask
        )
        if not np.any(reference_mask):
            raise ValueError("no valid pixels available for external reference")
        difference = hr_cube.data - reference.data
        sam_map = compute_sam(hr_cube.data, reference.data)
        sam_map = np.where(reference_mask, sam_map, np.nan)
        external_rmse = {}
        external_mae = {}
        for index, name in enumerate(hr_cube.band_names):
            values = difference[index][reference_mask]
            external_rmse[name] = float(np.sqrt(np.mean(values * values)))
            external_mae[name] = float(np.mean(np.abs(values)))
        mse = float(np.mean(difference[:, reference_mask] ** 2))
        psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
        ssim = _global_ssim(hr_cube.data, reference.data, reference_mask)
        means = np.mean(reference.data[:, reference_mask], axis=1)
        ergas = float(
            100.0
            * np.sqrt(
                np.mean(
                    (
                        np.asarray(list(external_rmse.values()))
                        / np.maximum(means, 1e-8)
                    )
                    ** 2
                )
            )
        )
        errors = external_rmse
        mae = external_mae

    return ValidationMetrics(
        sam_mean=float(np.nanmean(sam_map)),
        sam_map=sam_map,
        per_band_rmse=errors,
        per_band_mae=mae,
        psnr=psnr,
        ssim=ssim,
        ergas=ergas,
        reference_type=reference_type,
    )
