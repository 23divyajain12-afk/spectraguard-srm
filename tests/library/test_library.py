import numpy as np

from src.core.config import load_config
from src.core.schemas import RawSpectrum
from src.library.endmember_preparation import prepare_endmembers
from src.library.sensor_response import load_sensor_response
from src.library.usgs_io import load_usgs_spectrum


def test_library_loading_and_endmember_preparation(tmp_path):
    spectrum_path = tmp_path / "vegetation.txt"
    spectrum_path.write_text(
        "# name: vegetation\n# material_class: vegetation\n400 0.1\n500 0.2\n600 0.3\n700 0.4\n",
        encoding="utf-8",
    )
    response_path = tmp_path / "response.npz"
    wavelength = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)
    np.savez(response_path, wavelength_nm=wavelength, B02=np.ones(4), B03=np.ones(4) * 2)

    spectrum = load_usgs_spectrum(str(spectrum_path))
    response = load_sensor_response(str(response_path), ["B02", "B03"])
    endmembers = prepare_endmembers(
        [spectrum, RawSpectrum("soil", "soil", spectrum.wavelength_nm, spectrum.reflectance * 0.5, {})],
        response,
        load_config("config/default.yaml"),
    )
    assert endmembers.A.shape == (2, 2)
    assert np.all((endmembers.A >= 0.0) & (endmembers.A <= 1.0))
