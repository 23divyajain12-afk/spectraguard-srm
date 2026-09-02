import numpy as np

from src.core.config import RunConfig
from src.core.schemas import AbundanceResult, EndmemberSet, FusedCube


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the non-negative unit-sum simplex."""
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, len(vector) + 1)
    valid = sorted_values - (cumulative - 1.0) / indices > 0
    if not np.any(valid):
        return np.full(vector.shape, 1.0 / len(vector))
    rho = indices[valid][-1]
    threshold = (cumulative[rho - 1] - 1.0) / rho
    return np.maximum(vector - threshold, 0.0)


def fcls(
    hr_cube: FusedCube,
    endmembers: EndmemberSet,
    config: RunConfig,
) -> AbundanceResult:
    """Estimate abundances with non-negativity and an exact sum-to-one projection."""
    data = np.asarray(hr_cube.data, dtype=np.float64)
    matrix = np.asarray(endmembers.A, dtype=np.float64)
    if data.ndim != 3 or matrix.ndim != 2:
        raise ValueError("cube data and endmember matrix must be 3D and 2D")
    bands, height, width = data.shape
    if matrix.shape[0] != bands or matrix.shape[1] == 0:
        raise ValueError("endmember matrix must have one row per cube band")
    if len(endmembers.names) != matrix.shape[1]:
        raise ValueError("endmember names must match the matrix columns")
    if hr_cube.mask.shape != (height, width):
        raise ValueError("cube mask must match the spatial dimensions")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("endmember matrix contains non-finite values")

    gram = matrix.T @ matrix
    lipschitz = max(float(np.linalg.eigvalsh(gram).max()), 1e-12)
    step = 1.0 / lipschitz
    abundance = np.zeros((matrix.shape[1], height, width), dtype=np.float32)
    residual = np.full((height, width), np.nan, dtype=np.float32)
    for row, col in zip(*np.where(hr_cube.mask)):
        pixel = data[:, row, col]
        estimate = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        for _ in range(500):
            gradient = matrix.T @ (matrix @ estimate - pixel)
            updated = _project_simplex(estimate - step * gradient)
            if np.max(np.abs(updated - estimate)) < 1e-8:
                estimate = updated
                break
            estimate = updated
        abundance[:, row, col] = estimate.astype(np.float32)
        residual[row, col] = np.float32(np.linalg.norm(pixel - matrix @ estimate))

    return AbundanceResult(
        S=abundance,
        endmember_names=list(endmembers.names),
        residual=residual,
        solver="fcls",
    )
