from typing import Dict

import numpy as np

from src.core.schemas import SentinelCube


def compute_sam(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute spectral angle in radians for vectors or channel-first rasters."""
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 1:
        raise ValueError("x and y must have matching non-empty shapes")
    if np.array_equal(first, second):
        shape = first.shape[1:] if first.ndim > 1 else ()
        return np.zeros(shape, dtype=np.float64)
    if first.ndim == 1:
        axis = 0
    else:
        axis = 0
    numerator = np.sum(first * second, axis=axis)
    denominator = np.linalg.norm(first, axis=axis) * np.linalg.norm(second, axis=axis)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def compute_band_errors(pred: SentinelCube, ref: SentinelCube) -> Dict[str, float]:
    """Return mask-aware per-band RMSE values."""
    if pred.data.shape != ref.data.shape:
        raise ValueError("prediction and reference cubes must have matching shapes")
    if pred.band_names != ref.band_names:
        raise ValueError("prediction and reference bands must match in order")
    if pred.mask.shape != pred.data.shape[1:] or ref.mask.shape != ref.data.shape[1:]:
        raise ValueError("cube masks must match their spatial dimensions")
    mask = pred.mask & ref.mask
    mask &= np.all(np.isfinite(pred.data), axis=0) & np.all(np.isfinite(ref.data), axis=0)
    if not np.any(mask):
        raise ValueError("no valid pixels available for band errors")
    errors = {}
    for index, name in enumerate(pred.band_names):
        difference = pred.data[index][mask] - ref.data[index][mask]
        errors[name] = float(np.sqrt(np.mean(difference * difference)))
    return errors
