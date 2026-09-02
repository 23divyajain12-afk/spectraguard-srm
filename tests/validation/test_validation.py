import numpy as np

from src.consistency.degradation import degrade
from src.core.config import load_config
from src.core.schemas import FusedCube, ValidationMetrics
from src.validation.metrics import evaluate
from src.validation.spectral_metrics import compute_band_errors, compute_sam
from src.validation.uncertainty import estimate_uncertainty
from tests.fixtures.fixture_loader import load_small_aoi


def _hr_cube():
    cube, _ = load_small_aoi()
    data = np.repeat(np.repeat(cube.data, 4, axis=1), 4, axis=2)
    mask = np.repeat(np.repeat(cube.mask, 4, axis=0), 4, axis=1)
    return FusedCube(
        data=data,
        band_names=cube.band_names,
        crs=cube.crs,
        transform=cube.transform,
        resolution_m=2.5,
        bounds=cube.bounds,
        mask=mask,
        meta=cube.meta,
        alpha_map=np.ones_like(data, dtype=np.float32) * 0.1,
    )


def test_sam_and_band_errors():
    cube, _ = load_small_aoi()
    assert np.allclose(compute_sam(cube.data, cube.data), 0.0)
    assert compute_band_errors(cube, cube)["B02"] == 0.0


def test_evaluation_and_uncertainty_contracts():
    config = load_config("config/default.yaml")
    original, _ = load_small_aoi()
    hr_cube = _hr_cube()
    metrics = evaluate(hr_cube, original, None, config)
    assert isinstance(metrics, ValidationMetrics)
    assert metrics.reference_type == "internal_consistency"
    assert metrics.sam_map.shape == original.mask.shape

    uncertainty = estimate_uncertainty(hr_cube, metrics, config)
    assert uncertainty.U.shape == original.mask.shape
    assert np.all((uncertainty.U >= 0.0) & (uncertainty.U <= 1.0))
    assert set(uncertainty.components) == {"sam", "reconstruction", "detail"}
