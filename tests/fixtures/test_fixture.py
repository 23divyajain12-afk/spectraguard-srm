import numpy as np

from tests.fixtures.fixture_loader import load_small_aoi


def test_small_aoi_fixture_contract():
    cube, endmembers = load_small_aoi()

    assert cube.data.shape == (5, 64, 64)
    assert len(cube.band_names) == 5
    assert cube.data.dtype == np.float32
    assert np.all((cube.data >= 0.0) & (cube.data <= 1.0))
    assert cube.mask.shape == (64, 64)
    assert cube.mask.dtype == np.bool_
    assert cube.crs == "EPSG:32643"
    assert cube.resolution_m == 10.0
    assert cube.meta["scene_id"] == "synthetic-aoi-small"
    assert cube.meta["source"] == "deterministic synthetic fixture"

    assert endmembers.A.shape == (5, 3)
    assert len(endmembers.names) == 3
    assert len(endmembers.material_classes) == 3
    assert len(endmembers.source_spectra) == 3


def test_small_aoi_fixture_is_reproducible():
    first_cube, first_endmembers = load_small_aoi()
    second_cube, second_endmembers = load_small_aoi()

    np.testing.assert_array_equal(first_cube.data, second_cube.data)
    np.testing.assert_array_equal(first_cube.mask, second_cube.mask)
    np.testing.assert_array_equal(first_endmembers.A, second_endmembers.A)
