import numpy as np
import pytest

from src.core.config import load_config
from src.core.schemas import SentinelCube
from src.sr.model_adapter import Sen2SRModel, load_sr_model
from src.sr.spatial_representation import build_spatial_input


def test_sen2sr_selection_returns_valid_adapter():
    config = load_config("config/default.yaml")
    config.sr.model = "sen2sr"
    model = load_sr_model(config)

    assert isinstance(model, Sen2SRModel)
    assert model.model_name == "SEN2SR"
    assert callable(model.predict)


def test_sen2sr_predict_requires_rgb_nir_channels():
    from src.sr.model_adapter import Sen2SRModel

    adapter = object.__new__(Sen2SRModel)
    with pytest.raises(ValueError, match="exactly four channels"):
        adapter.predict(np.zeros((5, 8, 8), dtype=np.float32))


def test_sen2sr_spatial_input_excludes_b11():
    config = load_config("config/default.yaml")
    config.sr.model = "sen2sr"
    cube = SentinelCube(
        data=np.zeros((5, 4, 4), dtype=np.float32),
        band_names=["B02", "B03", "B04", "B08", "B11"],
        crs="EPSG:32643",
        transform=(10, 0, 0, 0, -10, 0),
        resolution_m=10,
        bounds=(0, 0, 40, 40),
        mask=np.ones((4, 4), dtype=bool),
    )
    representation = build_spatial_input(cube, config)
    assert representation.source_bands == ["B02", "B03", "B04", "B08"]
    assert representation.array.shape[0] == 4
