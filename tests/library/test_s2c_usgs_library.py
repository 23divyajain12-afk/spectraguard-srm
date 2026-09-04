import numpy as np
from pathlib import Path

from src.library.s2c_usgs_library import (
    load_sensor_response,
    load_usgs_spectrum,
    prepare_endmembers,
)

ROOT = Path(__file__).resolve().parents[2]
SRF = ROOT / "data/raw/sensor_response/COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"
MANIFEST = ROOT / "data/raw/spectral_library/endmember_manifest.json"
SPECTRA = ROOT / "data/raw/spectral_library/selected_usgs"


def test_s2c_srf_has_exact_five_bands():
    sr = load_sensor_response(str(SRF), ["B02", "B03", "B04", "B08", "B11"])
    assert sr.band_names == ["B02", "B03", "B04", "B08", "B11"]
    assert np.all(np.isfinite(sr.wavelength_nm))
    for b in sr.band_names:
        assert np.all(np.isfinite(sr.rsr[b]))
        assert np.all(sr.rsr[b] >= 0)
        assert np.any(sr.rsr[b] > 0)


def test_flat_spectrum_is_band_invariant():
    sr = load_sensor_response(str(SRF), ["B02", "B03", "B04", "B08", "B11"])
    x = np.arange(350.0, 2501.0)
    flat = np.full_like(x, 0.25)
    from src.library.s2c_usgs_library import _integrate_srf
    vals = [
        _integrate_srf(x, flat, sr.wavelength_nm, sr.rsr[b])
        for b in sr.band_names
    ]
    assert np.allclose(vals, 0.25, atol=1e-6)


def test_active_asdf_library_has_shape_5x6():
    files = [
        ("a500b9f6-7382-4188-81da-7293bf040bce.txt", "asphalt"),
        ("1d4e2b8c-1fe1-40e6-a45d-a7fade693fe0.txt", "concrete"),
        ("7d9aa0b5-d9c3-4242-a2de-16d31ea10e6e.txt", "roofing"),
        ("e6e039b4-94a2-4fbb-9de4-1e6f7a3f51ce.txt", "sand"),
        ("c8152390-b1ba-4d0b-9245-d7375a23dee3.txt", "metal"),
        ("9e857cbc-2003-42af-88e1-0554153a5a29.txt", "dry_vegetation"),
    ]
    library = [
        load_usgs_spectrum(str(SPECTRA / f), material_class=cls)
        for f, cls in files
    ]
    sr = load_sensor_response(str(SRF), ["B02", "B03", "B04", "B08", "B11"])
    em = prepare_endmembers(library, sr)
    assert em.A.shape == (5, 6)
    assert np.all(np.isfinite(em.A))
    assert np.all(em.A >= 0)


def test_bad_channels_are_not_propagated():
    spec = load_usgs_spectrum(
        str(SPECTRA / "e6e039b4-94a2-4fbb-9de4-1e6f7a3f51ce.txt"),
        material_class="sand",
    )
    assert np.all(np.isfinite(spec.reflectance))
    assert np.all(spec.reflectance >= 0)


def test_beck_spectrum_requires_exact_native_wavelengths():
    import pytest
    with pytest.raises(ValueError, match="exact USGS native wavelength record"):
        load_usgs_spectrum(
            str(SPECTRA / "f00dcb33-6e8d-4a85-8e28-25c3136a158f.txt"),
            material_class="vegetation",
        )
