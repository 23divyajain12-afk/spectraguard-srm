import numpy as np
import pytest

from src.core.config import load_config
from src.fusion.bicubic import bicubic_upscale
from src.fusion.detail_residual import extract_residual
from src.fusion.spectral_injection import fuse_multispectral, run_safety_checks
from tests.fixtures.fixture_loader import load_small_aoi


def test_bicubic_and_residual_contracts():
    cube, _ = load_small_aoi()
    baseline = bicubic_upscale(cube, 4)
    assert baseline.data.shape == (5, 256, 256)
    assert baseline.resolution_m == 2.5
    assert baseline.meta["upscale_method"] == "bicubic"

    residual = extract_residual(baseline.data + 0.01, baseline.data)
    assert residual.D.shape == baseline.data.shape
    np.testing.assert_allclose(residual.D, 0.01, atol=1e-7)


def test_spectral_injection_and_safety_checks():
    cube, _ = load_small_aoi()
    config = load_config("config/default.yaml")
    baseline = bicubic_upscale(cube, 4)
    spatial_base = baseline.data[:3]
    spatial_hr = spatial_base + 0.01
    fused = fuse_multispectral(baseline, spatial_base, spatial_hr, config)
    assert fused.data.shape == baseline.data.shape
    assert fused.alpha_map.shape == baseline.data.shape
    assert np.all((fused.data >= 0) & (fused.data <= 1))

    fused.data[0, 0, 0] = 2.0
    checked = run_safety_checks(fused, config)
    assert checked.anomaly_mask[0, 0]
    assert checked.data[0, 0, 0] == 1.0


def test_matching_sr_channels_are_fused_independently():
    cube, _ = load_small_aoi()
    config = load_config("config/default.yaml")
    baseline = bicubic_upscale(cube, 4)
    base = baseline.data.copy()
    detail = np.zeros_like(base)
    detail[0] = 0.01
    detail[1] = 0.02
    detail[2] = 0.03
    detail[3] = 0.04

    fused = fuse_multispectral(baseline, base, base + detail, config)

    expected = baseline.data + fused.alpha_map * detail
    expected[:, ~baseline.mask] = baseline.data[:, ~baseline.mask]
    np.testing.assert_allclose(fused.data, np.clip(expected, 0.0, 1.0), atol=1e-6)


def test_invalid_residual_shapes_fail():
    with pytest.raises(ValueError):
        extract_residual(np.zeros((2, 2)), np.zeros((3, 3)))
