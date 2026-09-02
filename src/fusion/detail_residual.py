import numpy as np

from src.core.schemas import SRResidual


def extract_residual(
    sr_output: np.ndarray, bicubic_base: np.ndarray
) -> SRResidual:
    """Return the spatial detail added by SR beyond the bicubic base."""
    sr = np.asarray(sr_output, dtype=np.float32)
    base = np.asarray(bicubic_base, dtype=np.float32)
    if sr.shape != base.shape:
        raise ValueError("sr_output and bicubic_base must have the same shape")
    if sr.ndim not in (2, 3):
        raise ValueError("SR arrays must have shape (H,W) or (C,H,W)")
    return SRResidual(
        D=sr - base,
        sr_output=sr,
        bicubic_base=base,
        model_name="unknown",
        model_version="unknown",
        scale=1,
    )
