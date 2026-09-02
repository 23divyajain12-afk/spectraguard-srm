import numpy as np

from src.consistency.degradation import degrade
from src.core.config import RunConfig
from src.core.schemas import FusedCube, UncertaintyMap, ValidationMetrics


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    y = np.linspace(0, array.shape[0] - 1, height)
    x = np.linspace(0, array.shape[1] - 1, width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, array.shape[0] - 1)
    x1 = np.minimum(x0 + 1, array.shape[1] - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = array[y0[:, None], x0[None, :]] * (1 - wx)
    top += array[y0[:, None], x1[None, :]] * wx
    bottom = array[y1[:, None], x0[None, :]] * (1 - wx)
    bottom += array[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def _normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    valid = mask & np.isfinite(values)
    if np.any(valid):
        low = values[valid].min()
        high = values[valid].max()
        if high > low:
            result[valid] = (values[valid] - low) / (high - low)
    return result


def estimate_uncertainty(
    hr_cube: FusedCube,
    metrics: ValidationMetrics,
    config: RunConfig,
) -> UncertaintyMap:
    """Combine SAM, reconstruction error, and injected-detail magnitude."""
    prediction = degrade(hr_cube, config).predicted
    if metrics.sam_map.shape != prediction.mask.shape:
        raise ValueError("metrics.sam_map must match the degraded spatial grid")
    reconstruction = np.mean(
        np.abs(prediction.data - np.mean(prediction.data, axis=(1, 2), keepdims=True)),
        axis=0,
    )
    if hr_cube.alpha_map is None:
        detail = np.zeros(hr_cube.data.shape[1:], dtype=np.float32)
    else:
        detail = np.mean(np.abs(hr_cube.alpha_map), axis=0)
    if detail.shape != prediction.mask.shape:
        detail = np.asarray(
            _resize(detail, prediction.mask.shape[0], prediction.mask.shape[1]),
            dtype=np.float32,
        )
    detail = _normalize(detail, prediction.mask)
    sam = _normalize(np.nan_to_num(metrics.sam_map, nan=0.0), prediction.mask)
    reconstruction = _normalize(reconstruction, prediction.mask)
    components = {"sam": sam, "reconstruction": reconstruction, "detail": detail}
    weights = {"sam": 1 / 3, "reconstruction": 1 / 3, "detail": 1 / 3}
    uncertainty = sum(weights[name] * value for name, value in components.items())
    return UncertaintyMap(
        U=np.clip(uncertainty, 0.0, 1.0).astype(np.float32),
        components=components,
        weights=weights,
    )
