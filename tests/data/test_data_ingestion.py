from pathlib import Path

import numpy as np

from src.core.config import load_config
from src.data.geospatial import crop_to_aoi, reproject_match
from src.data.preprocessing import preprocess
from src.data.sentinel_io import load_sentinel
from tests.fixtures.fixture_loader import load_small_aoi


ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "aoi_small.npz"
CONFIG = load_config(str(ROOT / "config" / "default.yaml"))


def test_load_sentinel_returns_contract():
    cube = load_sentinel(str(FIXTURE), CONFIG)
    assert cube.data.shape == (5, 64, 64)
    assert cube.data.dtype == np.float32
    assert cube.mask.shape == (64, 64)
    assert cube.crs == "EPSG:32643"


def test_preprocess_uses_working_grid_and_mask():
    cube, _ = load_small_aoi()
    processed = preprocess(cube, CONFIG)
    assert processed.resolution_m == CONFIG.data.working_resolution_m
    assert processed.data.shape == cube.data.shape
    assert processed.data.dtype == np.float32
    assert np.all((processed.data >= 0) & (processed.data <= 1))
    assert processed.mask.dtype == np.bool_
    assert processed.meta["resample_method"] == "bilinear"


def test_reproject_match_and_crop():
    cube, _ = load_small_aoi()
    target_transform = (20.0, 0.0, 500000.0, 0.0, -20.0, 4600000.0)
    reprojected = reproject_match(cube, "EPSG:32644", target_transform)
    assert reprojected.crs == "EPSG:32644"
    assert reprojected.data.shape == (5, 32, 32)
    assert reprojected.transform == target_transform

    cropped = crop_to_aoi(cube, (500100.0, 4599700.0, 500300.0, 4599900.0))
    assert cropped.data.shape == (5, 20, 20)
    assert cropped.bounds == (500100.0, 4599700.0, 500300.0, 4599900.0)


def test_invalid_scene_path_fails_loudly():
    try:
        load_sentinel(str(ROOT / "missing.npz"), CONFIG)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing scene should raise FileNotFoundError")
