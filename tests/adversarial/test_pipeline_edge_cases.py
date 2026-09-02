from pathlib import Path

import numpy as np
import pytest

from src.consistency.degradation import degrade
from src.consistency.projection import project_to_measurement_consistency
from src.core.config import load_config
from src.core.schemas import EndmemberSet, FusedCube, RawSpectrum, SentinelCube, ValidationMetrics
from src.data.geospatial import crop_to_aoi
from src.data.preprocessing import preprocess
from src.data.sentinel_io import load_sentinel
from src.fusion.bicubic import bicubic_upscale
from src.fusion.detail_residual import extract_residual
from src.fusion.spectral_injection import fuse_multispectral
from src.library.endmember_preparation import prepare_endmembers
from src.library.sensor_response import load_sensor_response
from src.library.usgs_io import load_usgs_spectrum
from src.sr.model_adapter import load_sr_model
from src.sr.spatial_representation import build_spatial_input
from src.sr.tiling import tiled_inference
from src.validation.metrics import evaluate
from src.validation.spectral_metrics import compute_band_errors, compute_sam
from tests.fixtures.fixture_loader import load_small_aoi


ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "aoi_small.npz"
CONFIG = load_config(str(ROOT / "config" / "default.yaml"))


def _cube(data=None, mask=None):
    source, _ = load_small_aoi()
    if data is None:
        data = source.data
    if mask is None:
        mask = source.mask
    return SentinelCube(
        data=np.asarray(data, dtype=np.float32),
        band_names=list(source.band_names),
        crs=source.crs,
        transform=source.transform,
        resolution_m=source.resolution_m,
        bounds=source.bounds,
        mask=np.asarray(mask, dtype=bool),
        meta=dict(source.meta),
    )


def _fused(scale=1, data=None, mask=None):
    source = _cube(data, mask)
    return FusedCube(
        data=source.data,
        band_names=source.band_names,
        crs=source.crs,
        transform=source.transform,
        resolution_m=source.resolution_m / scale,
        bounds=source.bounds,
        mask=source.mask,
        meta=source.meta,
    )


def test_load_sentinel_rejects_nonexistent_and_malformed_npz(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sentinel(str(tmp_path / "missing.npz"), CONFIG)

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, data=np.zeros((5, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        load_sentinel(str(malformed), CONFIG)


def test_load_sentinel_rejects_wrong_dimensions_and_nonfinite_data(tmp_path):
    with np.load(FIXTURE, allow_pickle=False) as values:
        fields = {name: values[name] for name in values.files}

    wrong_dimensions = tmp_path / "wrong_dimensions.npz"
    fields["data"] = np.zeros((64, 64), dtype=np.float32)
    np.savez(wrong_dimensions, **fields)
    with pytest.raises(ValueError, match="shape"):
        load_sentinel(str(wrong_dimensions), CONFIG)

    nonfinite = tmp_path / "nonfinite.npz"
    fields["data"] = np.zeros((5, 64, 64), dtype=np.float32)
    fields["data"][0, 0, 0] = np.nan
    np.savez(nonfinite, **fields)
    with pytest.raises(ValueError, match="non-finite"):
        load_sentinel(str(nonfinite), CONFIG)


def test_preprocess_rejects_empty_and_mismatched_masks():
    with pytest.raises(ValueError, match="shapes"):
        preprocess(_cube(data=np.empty((5, 0, 4)), mask=np.empty((0, 4))), CONFIG)

    source, _ = load_small_aoi()
    invalid = SentinelCube(
        data=source.data,
        band_names=source.band_names,
        crs=source.crs,
        transform=source.transform,
        resolution_m=source.resolution_m,
        bounds=source.bounds,
        mask=np.ones((63, 64), dtype=bool),
    )
    with pytest.raises(ValueError, match="shapes"):
        preprocess(invalid, CONFIG)


def test_crop_rejects_invalid_and_nonintersecting_aoi():
    cube = _cube()
    with pytest.raises(ValueError, match="AOI"):
        crop_to_aoi(cube, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="intersect"):
        crop_to_aoi(cube, (0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="Polygon"):
        crop_to_aoi(cube, {"type": "Point", "coordinates": [500100.0, 4599900.0]})


def test_tiling_handles_nondivisible_and_oversized_tiles():
    model = load_sr_model(CONFIG)
    image = np.ones((3, 10, 13), dtype=np.float32)
    nondivisible = tiled_inference(model, image, tile_size=6, overlap=2)
    oversized = tiled_inference(model, image, tile_size=32, overlap=4)
    assert nondivisible.shape == (3, 40, 52)
    assert oversized.shape == (3, 40, 52)
    np.testing.assert_allclose(nondivisible, 1.0)
    np.testing.assert_allclose(oversized, 1.0)


def test_invalid_sr_scale_tile_and_spatial_shapes_raise():
    with pytest.raises(ValueError, match="scale"):
        load_sr_model(type(CONFIG)(
            project=CONFIG.project,
            data=CONFIG.data,
            sr=type(CONFIG.sr)(CONFIG.sr.model, 0, CONFIG.sr.tile_size, CONFIG.sr.overlap),
            fusion=CONFIG.fusion,
            consistency=CONFIG.consistency,
            unmixing=CONFIG.unmixing,
        ))
    model = load_sr_model(CONFIG)
    with pytest.raises(ValueError, match="overlap"):
        tiled_inference(model, np.ones((8, 8), dtype=np.float32), 4, 4)
    with pytest.raises(ValueError, match="shape"):
        build_spatial_input(_cube(data=np.ones((5, 64))), CONFIG)


def test_fusion_rejects_mismatched_and_invalid_residual_shapes():
    cube = bicubic_upscale(_cube(), 2)
    with pytest.raises(ValueError, match="match"):
        fuse_multispectral(cube, np.zeros((16, 16)), np.zeros((32, 32)), CONFIG)
    with pytest.raises(ValueError, match="shape"):
        extract_residual(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))


def test_fusion_preserves_masked_pixel_values():
    cube = bicubic_upscale(_cube(), 2)
    mask = cube.mask.copy()
    mask[0, 0] = False
    cube.mask = mask
    base = cube.data[:3]
    fused = fuse_multispectral(cube, base, base + 0.2, CONFIG)
    np.testing.assert_array_equal(fused.data[:, 0, 0], cube.data[:, 0, 0])


def test_consistency_and_validation_reject_shape_mismatches():
    high_resolution = _fused()
    original = _cube(data=np.zeros((4, 64, 64)))
    with pytest.raises(ValueError, match="band count"):
        project_to_measurement_consistency(high_resolution, original, CONFIG)

    with pytest.raises(ValueError, match="matching shapes"):
        compute_band_errors(_cube(), _cube(data=np.zeros((5, 32, 32))))

    with pytest.raises(ValueError, match="matching"):
        compute_sam(np.zeros((3, 2)), np.zeros((4, 2)))


def test_sam_identical_spectra_is_exactly_zero_and_zero_spectrum_is_finite():
    spectra = np.array([[0.2, 0.4], [0.5, 0.1]], dtype=np.float32)
    np.testing.assert_array_equal(compute_sam(spectra, spectra), np.zeros(2))
    zero_result = compute_sam(np.zeros(3), np.ones(3))
    assert np.isfinite(zero_result)
    assert zero_result == 0.0


def test_library_rejects_malformed_duplicate_and_missing_sensor_data(tmp_path):
    malformed = tmp_path / "malformed.txt"
    malformed.write_text("not a spectrum\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two numeric"):
        load_usgs_spectrum(str(malformed))

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("400 0.1\n500 0.2\n500 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_usgs_spectrum(str(duplicate))

    response = tmp_path / "response.npz"
    np.savez(response, wavelength_nm=np.array([400.0, 500.0]), B02=np.ones(2))
    with pytest.raises(ValueError, match="missing"):
        load_sensor_response(str(response), ["B02", "B03"])


def test_library_sorts_unsorted_wavelengths_and_rejects_insufficient_coverage(tmp_path):
    unsorted = tmp_path / "unsorted.txt"
    unsorted.write_text("500 0.2\n400 0.1\n600 0.3\n", encoding="utf-8")
    spectrum = load_usgs_spectrum(str(unsorted))
    np.testing.assert_array_equal(spectrum.wavelength_nm, [400.0, 500.0, 600.0])

    response = tmp_path / "response.npz"
    np.savez(response, wavelength_nm=np.array([450.0, 500.0, 550.0]), B02=np.ones(3))
    sensor = load_sensor_response(str(response), ["B02"])
    with pytest.raises(ValueError, match="coverage|range|wavelength"):
        prepare_endmembers(
            [RawSpectrum("partial", "synthetic", np.array([480.0, 520.0]), np.array([0.2, 0.3]), {})],
            sensor,
            CONFIG,
        )
