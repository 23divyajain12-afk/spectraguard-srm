import numpy as np

from src.consistency.degradation import degrade
from src.consistency.projection import project_to_measurement_consistency
from src.core.config import load_config
from src.core.schemas import FusedCube
from tests.fixtures.fixture_loader import load_small_aoi


def _hr_cube():
    cube, _ = load_small_aoi()
    scale = 4
    data = np.repeat(np.repeat(cube.data, scale, axis=1), scale, axis=2)
    mask = np.repeat(np.repeat(cube.mask, scale, axis=0), scale, axis=1)
    return FusedCube(
        data=data,
        band_names=cube.band_names,
        crs=cube.crs,
        transform=cube.transform,
        resolution_m=cube.resolution_m / scale,
        bounds=cube.bounds,
        mask=mask,
        meta=cube.meta,
    )


def test_degrade_returns_original_grid_shape():
    config = load_config("config/default.yaml")
    prediction = degrade(_hr_cube(), config)
    assert prediction.predicted.data.shape == (5, 64, 64)
    assert prediction.degradation_method == "bilinear_downsample"


def test_projection_preserves_contract_and_corrects_values():
    config = load_config("config/default.yaml")
    high_res = _hr_cube()
    high_res.data = np.clip(high_res.data + 0.1, 0.0, 1.0)
    projected = project_to_measurement_consistency(high_res, load_small_aoi()[0], config)
    assert projected.data.shape == high_res.data.shape
    assert projected.data.dtype == np.float32
    assert np.all((projected.data >= 0.0) & (projected.data <= 1.0))
