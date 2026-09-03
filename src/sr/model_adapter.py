import numpy as np

from src.core.config import RunConfig
from src.core.schemas import SRModel


class NumpyInterpolationModel:
    """Small dependency-free SR adapter used when no external model is configured."""

    def __init__(self, scale: int, model_name: str) -> None:
        if scale < 1:
            raise ValueError("SR scale must be at least 1")
        self.scale = scale
        self.model_name = model_name

    def predict(self, spatial_input: np.ndarray) -> np.ndarray:
        array = np.asarray(spatial_input, dtype=np.float32)
        if array.ndim not in (2, 3):
            raise ValueError("spatial_input must have shape (H,W) or (C,H,W)")
        if array.shape[-2] == 0 or array.shape[-1] == 0:
            raise ValueError("spatial_input cannot be empty")
        return _resize(array, array.shape[-2] * self.scale, array.shape[-1] * self.scale)


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
    result = (top * (1 - wy) + bottom * wy).astype(np.float32)
    return result[0] if array.ndim == 2 else result


def load_sr_model(config: RunConfig) -> SRModel:
    """Load the configured replaceable SR model adapter."""
    if not config.sr.model:
        raise ValueError("SR model name must not be empty")
    if config.sr.model not in {"numpy_baseline", "numpy_interpolation"}:
        raise ValueError(
            f"SR model backend '{config.sr.model}' is unavailable; "
            "add its adapter and weights, or select 'numpy_baseline'"
        )
    return NumpyInterpolationModel(config.sr.scale, config.sr.model)
