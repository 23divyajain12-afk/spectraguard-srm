from typing import Optional

import numpy as np

from src.consistency.degradation import degrade
from src.core.config import RunConfig
from src.core.schemas import FusedCube, SentinelCube, ValidationMetrics
from src.validation.spectral_metrics import compute_band_errors, compute_sam


def evaluate(
    hr_cube: FusedCube,
    original_cube: SentinelCube,
    reference: Optional[SentinelCube],
    config: RunConfig,
) -> ValidationMetrics:
    """Evaluate the degraded HR result against the original observation."""
    predicted = degrade(hr_cube, config).predicted
    if predicted.data.shape != original_cube.data.shape:
        raise ValueError("degraded prediction and original cube must align")
    mask = predicted.mask & original_cube.mask
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
        if reference.data.shape != hr_cube.data.shape:
            raise ValueError("reference must match the HR cube dimensions")
        reference_type = str(reference.meta.get("reference_type", "reference_imagery"))
        difference = hr_cube.data - reference.data
        mse = float(np.mean(difference * difference))
        psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
        rmse = float(np.sqrt(mse))
        ssim = float(max(0.0, 1.0 - rmse))
        means = np.mean(reference.data, axis=(1, 2))
        ergas = float(100.0 * np.sqrt(np.mean((errors_for_hr := np.sqrt(np.mean(difference * difference, axis=(1, 2))) / np.maximum(means, 1e-8)) ** 2)))

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
