import numpy as np
import pytest

from src.core.config import load_config
from src.sr.model_adapter import Sen2SRModel, load_sr_model


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
