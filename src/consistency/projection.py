import numpy as np

from src.consistency.degradation import _resize, degrade
from src.core.config import RunConfig
from src.core.schemas import FusedCube, SentinelCube


def project_to_measurement_consistency(
    hr_cube: FusedCube,
    original_cube: SentinelCube,
    config: RunConfig,
) -> FusedCube:
    """Iteratively correct HR data toward the measured Sentinel observation."""
    if hr_cube.data.shape[0] != original_cube.data.shape[0]:
        raise ValueError("HR and original cubes must have the same band count")
    if config.consistency.iterations < 0:
        raise ValueError("consistency iterations cannot be negative")
    if not 0.0 <= config.consistency.lambda_ <= 1.0:
        raise ValueError("consistency lambda must be between 0 and 1")

    data = hr_cube.data.astype(np.float32, copy=True)
    for _ in range(config.consistency.iterations):
        current = FusedCube(
            data=data,
            band_names=list(hr_cube.band_names),
            crs=hr_cube.crs,
            transform=hr_cube.transform,
            resolution_m=hr_cube.resolution_m,
            bounds=hr_cube.bounds,
            mask=hr_cube.mask,
            nodata=hr_cube.nodata,
            acquisition_time=hr_cube.acquisition_time,
            meta=dict(hr_cube.meta),
            alpha_map=hr_cube.alpha_map,
            provenance=hr_cube.provenance,
            anomaly_mask=hr_cube.anomaly_mask,
        )
        prediction = degrade(current, config).predicted
        residual = original_cube.data - prediction.data
        correction = _resize(residual, data.shape[1], data.shape[2])
        valid = _resize(
            (original_cube.mask & prediction.mask).astype(np.float32),
            data.shape[1],
            data.shape[2],
        )
        correction *= valid[None, ...]
        data = np.clip(data + config.consistency.lambda_ * correction, 0.0, 1.0)

    return FusedCube(
        data=data.astype(np.float32),
        band_names=list(hr_cube.band_names),
        crs=hr_cube.crs,
        transform=hr_cube.transform,
        resolution_m=hr_cube.resolution_m,
        bounds=hr_cube.bounds,
        mask=hr_cube.mask.copy(),
        nodata=hr_cube.nodata,
        acquisition_time=hr_cube.acquisition_time,
        meta={**hr_cube.meta, "consistency_iterations": config.consistency.iterations},
        alpha_map=hr_cube.alpha_map,
        provenance=hr_cube.provenance,
        anomaly_mask=hr_cube.anomaly_mask,
    )
