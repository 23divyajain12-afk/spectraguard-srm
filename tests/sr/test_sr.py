import numpy as np
import pytest

from src.core.config import load_config
from src.sr.model_adapter import load_sr_model
from src.sr.spatial_representation import build_spatial_input
from src.sr.tiling import tiled_inference
from tests.fixtures.fixture_loader import load_small_aoi


def test_spatial_representation_and_model():
    cube, _ = load_small_aoi()
    config = load_config("config/default.yaml")
    config.sr.model = "numpy_baseline"
    config.data.bands = []
    representation = build_spatial_input(cube, config)
    assert representation.array.shape == (3, 64, 64)
    assert representation.method == "pseudo_rgb"
    assert np.all((representation.array >= 0) & (representation.array <= 1))

    model = load_sr_model(config)
    output = model.predict(representation.array)
    assert output.shape == (3, 256, 256)
    assert output.dtype == np.float32


def test_tiled_inference_matches_expected_shape():
    config = load_config("config/default.yaml")
    config.sr.model = "numpy_baseline"
    model = load_sr_model(config)
    image = np.ones((3, 64, 64), dtype=np.float32)
    output = tiled_inference(model, image, tile_size=32, overlap=8)
    assert output.shape == (3, 256, 256)
    np.testing.assert_allclose(output, 1.0)


def test_invalid_tiling_arguments_fail():
    config = load_config("config/default.yaml")
    config.sr.model = "numpy_baseline"
    model = load_sr_model(config)
    with pytest.raises(ValueError):
        tiled_inference(model, np.ones((8, 8), dtype=np.float32), 8, 8)
