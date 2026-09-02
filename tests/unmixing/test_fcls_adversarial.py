from pathlib import Path

import numpy as np
import pytest

from src.core.config import load_config
from src.core.schemas import EndmemberSet, FusedCube
from src.unmixing.fcls import fcls


CONFIG_PATH = Path(__file__).parents[2] / "config" / "default.yaml"


def _config():
    return load_config(str(CONFIG_PATH))


def _cube(pixels: np.ndarray, mask: np.ndarray | None = None) -> FusedCube:
    """Build a channel-first cube from a (pixels, bands) array."""
    pixel_count, band_count = pixels.shape
    if mask is None:
        mask = np.ones((1, pixel_count), dtype=bool)
    return FusedCube(
        data=pixels.T.reshape(band_count, 1, pixel_count).astype(np.float32),
        band_names=[f"B{index + 1}" for index in range(band_count)],
        crs="EPSG:32643",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        resolution_m=1.0,
        bounds=(0.0, 0.0, float(pixel_count), 1.0),
        mask=mask,
    )


def _endmembers(matrix: np.ndarray) -> EndmemberSet:
    return EndmemberSet(
        A=matrix.T.astype(np.float32),
        names=["A", "B"],
        material_classes=["synthetic_a", "synthetic_b"],
        source_spectra=["synthetic_a", "synthetic_b"],
    )


def test_known_seventy_thirty_mixture_is_recovered():
    matrix = np.array([[0.8, 0.1, 0.2], [0.2, 0.7, 0.3]])
    observation = 0.7 * matrix[0] + 0.3 * matrix[1]

    result = fcls(_cube(observation[None, :]), _endmembers(matrix), _config())

    np.testing.assert_allclose(result.S[:, 0, 0], [0.7, 0.3], atol=1e-5)


def test_pure_endmember_pixels_recover_one_hot_abundances():
    matrix = np.array([[0.9, 0.05, 0.1], [0.1, 0.8, 0.4]])
    observations = np.vstack([matrix[0], matrix[1]])

    result = fcls(_cube(observations), _endmembers(matrix), _config())

    np.testing.assert_allclose(result.S[:, 0, 0], [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(result.S[:, 0, 1], [0.0, 1.0], atol=1e-5)


def test_multiple_known_mixtures_match_expected_abundance_matrix():
    matrix = np.array([[0.75, 0.10, 0.20], [0.15, 0.70, 0.35]])
    expected = np.array([[0.1, 0.4, 0.7, 0.95], [0.9, 0.6, 0.3, 0.05]])
    observations = expected.T @ matrix

    result = fcls(_cube(observations), _endmembers(matrix), _config())

    np.testing.assert_allclose(result.S[:, 0, :], expected, atol=1e-5)


def test_masked_pixels_are_zero_and_valid_pixels_remain_simplex_feasible():
    matrix = np.array([[0.8, 0.2, 0.1], [0.1, 0.7, 0.5]])
    observations = np.vstack(
        [
            0.25 * matrix[0] + 0.75 * matrix[1],
            0.6 * matrix[0] + 0.4 * matrix[1],
        ]
    )
    mask = np.array([[True, False]])

    result = fcls(_cube(observations, mask), _endmembers(matrix), _config())

    np.testing.assert_allclose(result.S[:, 0, 0], [0.25, 0.75], atol=1e-5)
    np.testing.assert_array_equal(result.S[:, 0, 1], [0.0, 0.0])
    assert np.isnan(result.residual[0, 1])


def test_incompatible_band_dimensions_raise_explicit_error():
    cube = _cube(np.array([[0.7, 0.3, 0.0]]))
    incompatible = EndmemberSet(
        A=np.eye(2, dtype=np.float32),
        names=["A", "B"],
        material_classes=["A", "B"],
        source_spectra=["A", "B"],
    )

    with pytest.raises(ValueError, match="one row per cube band"):
        fcls(cube, incompatible, _config())
