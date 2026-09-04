from pathlib import Path
from typing import List

import numpy as np

from src.core.schemas import SensorResponse


def load_sensor_response(path: str, band_names: List[str]) -> SensorResponse:
    """Load wavelength and relative-response columns from CSV or NPZ."""
    if not band_names:
        raise ValueError("band_names must not be empty")
    response_path = Path(path)
    if not response_path.is_file():
        raise FileNotFoundError(f"Sensor response file does not exist: {response_path}")
    if response_path.suffix.lower() in {".xlsx", ".xlsm"}:
        from src.library.s2c_usgs_library import load_sensor_response as load_s2c_response

        return load_s2c_response(str(response_path), band_names)

    if response_path.suffix.lower() == ".npz":
        with np.load(response_path, allow_pickle=False) as values:
            if "wavelength_nm" not in values.files:
                raise ValueError("Sensor response NPZ requires wavelength_nm")
            wavelength = np.asarray(values["wavelength_nm"], dtype=np.float32)
            rsr = {band: np.asarray(values[band], dtype=np.float32) for band in band_names if band in values.files}
    else:
        rows = np.loadtxt(response_path, delimiter=",", comments="#", skiprows=1)
        if rows.ndim != 2 or rows.shape[1] < len(band_names) + 1:
            raise ValueError("Sensor response CSV must contain wavelength and one column per band")
        wavelength = rows[:, 0].astype(np.float32)
        rsr = {band: rows[:, index + 1].astype(np.float32) for index, band in enumerate(band_names)}

    if len(wavelength) < 2 or any(band not in rsr for band in band_names):
        raise ValueError("Sensor response is missing required band curves")
    if any(curve.shape != wavelength.shape for curve in rsr.values()):
        raise ValueError("All response curves must match the wavelength grid")
    if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Sensor wavelengths must be finite and strictly increasing")
    return SensorResponse(
        band_names=list(band_names),
        wavelength_nm=wavelength,
        rsr={band: np.clip(rsr[band], 0.0, None) for band in band_names},
    )
