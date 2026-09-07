import numpy as np

from src.core.config import RunConfig
from src.core.schemas import FusedCube, SentinelCube


def _spatial_channel(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array.mean(axis=0)
    raise ValueError("spatial arrays must have shape (H,W) or (C,H,W)")


def fuse_multispectral(
    bicubic_cube: SentinelCube,
    spatial_base: np.ndarray,
    spatial_hr: np.ndarray,
    config: RunConfig,
) -> FusedCube:
    """Inject controlled spatial detail into the bicubic multispectral cube."""
    base_array = np.asarray(spatial_base, dtype=np.float32)
    high_array = np.asarray(spatial_hr, dtype=np.float32)
    independent = (
        base_array.ndim == 3
        and high_array.ndim == 3
        and base_array.shape == high_array.shape
        and base_array.shape[0] == bicubic_cube.data.shape[0]
    )
    if independent:
        if base_array.shape[1:] != bicubic_cube.data.shape[1:]:
            raise ValueError("spatial inputs must match the bicubic cube grid")
        base = base_array
        high_resolution = high_array
    else:
        base = _spatial_channel(base_array)
        high_resolution = _spatial_channel(high_array)
    if base.shape[-2:] != high_resolution.shape[-2:] or base.shape[-2:] != bicubic_cube.data.shape[1:]:
        raise ValueError("spatial inputs must match the bicubic cube grid")
    detail = high_resolution - base
    if independent:
        variance = np.var(base, axis=(1, 2))
        alpha = np.array(
            [
                np.cov(band.ravel(), base[index].ravel(), bias=True)[0, 1]
                / (variance[index] + config.fusion.epsilon)
                if variance[index] > 0
                else 0.0
                for index, band in enumerate(bicubic_cube.data)
            ],
            dtype=np.float32,
        )
    else:
        variance = float(np.var(base))
        if variance <= 0:
            alpha = np.zeros(bicubic_cube.data.shape[0], dtype=np.float32)
        else:
            alpha = np.array(
                [np.cov(band.ravel(), base.ravel(), bias=True)[0, 1] / (variance + config.fusion.epsilon)
                 for band in bicubic_cube.data],
                dtype=np.float32,
            )
    alpha = np.clip(alpha, config.fusion.alpha_min, config.fusion.alpha_max)
    data = bicubic_cube.data + alpha[:, None, None] * detail
    data[:, ~bicubic_cube.mask] = bicubic_cube.data[:, ~bicubic_cube.mask]
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    alpha_map = np.broadcast_to(alpha[:, None, None], data.shape).copy()
    return FusedCube(
        data=data,
        band_names=list(bicubic_cube.band_names),
        crs=bicubic_cube.crs,
        transform=bicubic_cube.transform,
        resolution_m=bicubic_cube.resolution_m,
        bounds=bicubic_cube.bounds,
        mask=bicubic_cube.mask.copy(),
        nodata=bicubic_cube.nodata,
        acquisition_time=bicubic_cube.acquisition_time,
        meta=dict(bicubic_cube.meta),
        alpha_map=alpha_map,
    )


def run_safety_checks(fused: FusedCube, config: RunConfig) -> FusedCube:
    """Clip reflectance and mark pixels with invalid or out-of-range values."""
    invalid = ~np.isfinite(fused.data) | (fused.data < 0.0) | (fused.data > 1.0)
    anomaly_mask = np.any(invalid, axis=0)
    data = np.nan_to_num(fused.data, nan=0.0, posinf=1.0, neginf=0.0)
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    return FusedCube(
        data=data,
        band_names=list(fused.band_names),
        crs=fused.crs,
        transform=fused.transform,
        resolution_m=fused.resolution_m,
        bounds=fused.bounds,
        mask=fused.mask.copy(),
        nodata=fused.nodata,
        acquisition_time=fused.acquisition_time,
        meta=dict(fused.meta),
        alpha_map=fused.alpha_map,
        provenance=fused.provenance,
        anomaly_mask=anomaly_mask,
    )
