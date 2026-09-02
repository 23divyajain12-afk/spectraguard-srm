import numpy as np

from src.core.config import RunConfig
from src.core.schemas import SentinelCube


def _resize_band(band: np.ndarray, height: int, width: int) -> np.ndarray:
    if band.shape == (height, width):
        return band.astype(np.float32, copy=True)
    y = np.linspace(0, band.shape[0] - 1, height)
    x = np.linspace(0, band.shape[1] - 1, width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, band.shape[0] - 1)
    x1 = np.minimum(x0 + 1, band.shape[1] - 1)
    y_weight = (y - y0)[:, None]
    x_weight = (x - x0)[None, :]
    top = band[y0[:, None], x0[None, :]] * (1 - x_weight)
    top += band[y0[:, None], x1[None, :]] * x_weight
    bottom = band[y1[:, None], x0[None, :]] * (1 - x_weight)
    bottom += band[y1[:, None], x1[None, :]] * x_weight
    return (top * (1 - y_weight) + bottom * y_weight).astype(np.float32)


def preprocess(cube: SentinelCube, config: RunConfig) -> SentinelCube:
    """Align bands to the configured working grid and merge quality masks."""
    if cube.data.ndim != 3 or cube.mask.shape != cube.data.shape[1:]:
        raise ValueError("SentinelCube has inconsistent data and mask shapes")
    if cube.resolution_m <= 0 or config.data.working_resolution_m <= 0:
        raise ValueError("Raster resolutions must be positive")

    scale = cube.resolution_m / config.data.working_resolution_m
    height = max(1, round(cube.data.shape[1] * scale))
    width = max(1, round(cube.data.shape[2] * scale))
    data = np.stack([_resize_band(band, height, width) for band in cube.data])
    mask = _resize_band(cube.mask.astype(np.float32), height, width) >= 0.5

    for key in ("cloud_mask", "shadow_mask", "quality_mask"):
        if key in cube.meta:
            quality = np.asarray(cube.meta[key], dtype=bool)
            if quality.shape != cube.mask.shape:
                raise ValueError(f"{key} must match the source mask shape")
            if key == "quality_mask":
                mask &= _resize_band(quality.astype(np.float32), height, width) >= 0.5
            else:
                mask &= ~(_resize_band(quality.astype(np.float32), height, width) >= 0.5)

    metadata = dict(cube.meta)
    metadata["working_resolution_m"] = config.data.working_resolution_m
    metadata["resample_method"] = "bilinear"
    return SentinelCube(
        data=np.clip(data, 0.0, 1.0).astype(np.float32),
        band_names=list(cube.band_names),
        crs=cube.crs,
        transform=cube.transform,
        resolution_m=config.data.working_resolution_m,
        bounds=cube.bounds,
        mask=mask,
        nodata=cube.nodata,
        acquisition_time=cube.acquisition_time,
        meta=metadata,
    )
