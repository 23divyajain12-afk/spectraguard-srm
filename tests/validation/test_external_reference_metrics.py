import numpy as np
import pytest
from dataclasses import replace

from src.core.config import load_config
from src.core.schemas import FusedCube, SentinelCube
from src.validation.metrics import evaluate


def _cube(data, *, reference_type=None, mask=None, bands=None, crs="EPSG:32643"):
    data = np.asarray(data, dtype=np.float32)
    if mask is None:
        mask = np.ones(data.shape[1:], dtype=bool)
    meta = {} if reference_type is None else {"reference_type": reference_type}
    return FusedCube(
        data=data,
        band_names=bands or ["B02", "B03"],
        crs=crs,
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0),
        resolution_m=10.0,
        bounds=(500000.0, 4599960.0, 500040.0, 4600000.0),
        mask=np.asarray(mask, dtype=bool),
        meta=meta,
    )


def test_external_reference_metrics_use_reference_and_label():
    config = load_config("config/default.yaml")
    config = replace(config, sr=replace(config.sr, scale=1))
    prediction = _cube([[[0.6, 0.6], [0.6, 0.6]], [[0.3, 0.3], [0.3, 0.3]]])
    reference = _cube(
        [[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.3], [0.3, 0.3]]],
        reference_type="external_reference",
    )

    metrics = evaluate(prediction, prediction, reference, config)

    assert metrics.reference_type == "external_reference"
    assert metrics.per_band_rmse["B02"] == pytest.approx(0.1, abs=1e-6)
    assert metrics.per_band_mae["B02"] == pytest.approx(0.1, abs=1e-6)
    assert metrics.psnr is not None
    assert metrics.ssim is not None
    assert metrics.ergas is not None


def test_external_reference_sam_uses_high_resolution_reference():
    config = load_config("config/default.yaml")
    config = replace(config, sr=replace(config.sr, scale=1))
    degraded_observation = _cube(
        [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    prediction = _cube(
        [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    reference = _cube(
        [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
        reference_type="external_reference",
    )

    metrics = evaluate(prediction, degraded_observation, reference, config)

    assert metrics.sam_mean == pytest.approx(0.0, abs=1e-7)


def test_external_reference_requires_matching_grid_bands_and_crs():
    config = load_config("config/default.yaml")
    config = replace(config, sr=replace(config.sr, scale=1))
    prediction = _cube(np.ones((2, 2, 2), dtype=np.float32))
    reference = _cube(
        np.ones((2, 2, 2), dtype=np.float32),
        bands=["B03", "B02"],
    )

    with pytest.raises(ValueError, match="bands"):
        evaluate(prediction, prediction, reference, config)

    reference = _cube(np.ones((2, 2, 2), dtype=np.float32), crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS"):
        evaluate(prediction, prediction, reference, config)


def test_external_reference_mask_excludes_invalid_values():
    config = load_config("config/default.yaml")
    config = replace(config, sr=replace(config.sr, scale=1))
    mask = np.array([[True, False], [True, True]])
    prediction = _cube(
        [[[0.5, np.nan], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        mask=mask,
    )
    reference = _cube(
        [[[0.5, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        mask=mask,
        reference_type="synthetic_benchmark",
    )

    metrics = evaluate(prediction, prediction, reference, config)

    assert metrics.reference_type == "synthetic_benchmark"
    assert metrics.per_band_rmse["B02"] == pytest.approx(0.0)
