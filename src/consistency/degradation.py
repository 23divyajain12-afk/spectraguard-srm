import numpy as np

from src.core.config import RunConfig
from src.core.schemas import DegradationPrediction, FusedCube, SentinelCube


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    channels = array[None, ...] if array.ndim == 2 else array
    y = np.linspace(0, channels.shape[1] - 1, height)
    x = np.linspace(0, channels.shape[2] - 1, width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, channels.shape[1] - 1)
    x1 = np.minimum(x0 + 1, channels.shape[2] - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = channels[:, y0[:, None], x0[None, :]] * (1 - wx)
    top += channels[:, y0[:, None], x1[None, :]] * wx
    bottom = channels[:, y1[:, None], x0[None, :]] * (1 - wx)
    bottom += channels[:, y1[:, None], x1[None, :]] * wx
    result = top * (1 - wy) + bottom * wy
    return result[0] if array.ndim == 2 else result


def degrade(hr_cube: FusedCube, config: RunConfig) -> DegradationPrediction:
    """Approximate the Sentinel observation by blur-free area downsampling."""
    if hr_cube.data.ndim != 3 or hr_cube.mask.shape != hr_cube.data.shape[1:]:
        raise ValueError("hr_cube has inconsistent data and mask shapes")
    if config.sr.scale < 1:
        raise ValueError("SR scale must be at least 1")
    height = max(1, round(hr_cube.data.shape[1] / config.sr.scale))
    width = max(1, round(hr_cube.data.shape[2] / config.sr.scale))
    data = _resize(hr_cube.data, height, width).astype(np.float32)
    mask = _resize(hr_cube.mask.astype(np.float32), height, width) >= 0.5
    transform = list(hr_cube.transform)
    transform[0] *= config.sr.scale
    transform[4] *= config.sr.scale
    predicted = SentinelCube(
        data=np.clip(data, 0.0, 1.0),
        band_names=list(hr_cube.band_names),
        crs=hr_cube.crs,
        transform=tuple(transform),
        resolution_m=hr_cube.resolution_m * config.sr.scale,
        bounds=hr_cube.bounds,
        mask=mask,
        nodata=hr_cube.nodata,
        acquisition_time=hr_cube.acquisition_time,
        meta={**hr_cube.meta, "degradation_method": "bilinear_downsample"},
    )
    return DegradationPrediction(
        predicted=predicted,
        degradation_method="bilinear_downsample",
        kernel_params={"scale": config.sr.scale},
    )
