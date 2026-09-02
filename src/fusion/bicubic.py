import numpy as np

from src.core.schemas import SentinelCube
from src.sr.model_adapter import _resize


def bicubic_upscale(cube: SentinelCube, scale: int) -> SentinelCube:
    """Create the deterministic upscaled baseline for every spectral band."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if cube.data.ndim != 3 or cube.mask.shape != cube.data.shape[1:]:
        raise ValueError("cube has inconsistent data and mask shapes")
    data = _resize(cube.data, cube.data.shape[1] * scale, cube.data.shape[2] * scale)
    mask = _resize(
        cube.mask.astype(np.float32),
        cube.mask.shape[0] * scale,
        cube.mask.shape[1] * scale,
    ) >= 0.5
    transform = list(cube.transform)
    transform[0] /= scale
    transform[4] /= scale
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    bounds = (
        cube.bounds[0],
        cube.bounds[1],
        cube.bounds[0] + data.shape[2] * pixel_width,
        cube.bounds[1] + data.shape[1] * pixel_height,
    )
    metadata = dict(cube.meta)
    metadata["upscale_method"] = "bicubic"
    metadata["scale"] = scale
    return SentinelCube(
        data=np.clip(data, 0.0, 1.0),
        band_names=list(cube.band_names),
        crs=cube.crs,
        transform=tuple(transform),
        resolution_m=cube.resolution_m / scale,
        bounds=bounds,
        mask=mask,
        nodata=cube.nodata,
        acquisition_time=cube.acquisition_time,
        meta=metadata,
    )
