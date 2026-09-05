from typing import List

import numpy as np

from src.core.config import RunConfig
from src.core.schemas import EndmemberSet, RawSpectrum, SensorResponse


SEN2SR_BANDS = ["B02", "B03", "B04", "B08"]


def select_endmember_bands(
    endmembers: EndmemberSet,
    source_band_names: List[str],
    selected_band_names: List[str] = SEN2SR_BANDS,
) -> EndmemberSet:
    """Return an ordered band view without changing the original endmembers."""
    if endmembers.A.ndim != 2:
        raise ValueError("endmember matrix must be two-dimensional")
    if endmembers.A.shape[0] != len(source_band_names):
        raise ValueError("source band names must match the endmember matrix rows")
    if len(set(source_band_names)) != len(source_band_names):
        raise ValueError("source band names must be unique")
    if len(set(selected_band_names)) != len(selected_band_names):
        raise ValueError("selected band names must be unique")
    missing = [band for band in selected_band_names if band not in source_band_names]
    if missing:
        raise ValueError(f"selected bands are missing from endmembers: {missing}")
    indices = [source_band_names.index(band) for band in selected_band_names]
    return EndmemberSet(
        A=endmembers.A[indices].copy(),
        names=list(endmembers.names),
        material_classes=list(endmembers.material_classes),
        source_spectra=list(endmembers.source_spectra),
    )


def prepare_endmembers(
    library: List[RawSpectrum],
    sensor_response: SensorResponse,
    config: RunConfig,
) -> EndmemberSet:
    """Interpolate library spectra and integrate them through sensor RSR curves."""
    if not library:
        raise ValueError("library must contain at least one spectrum")
    grid = sensor_response.wavelength_nm
    columns = []
    for spectrum in library:
        wavelengths = np.asarray(spectrum.wavelength_nm, dtype=float)
        reflectance = np.asarray(spectrum.reflectance, dtype=float)
        if wavelengths.ndim != 1 or reflectance.shape != wavelengths.shape:
            raise ValueError(f"Invalid wavelength/reflectance shape for {spectrum.name}")
        valid = np.isfinite(wavelengths) & np.isfinite(reflectance)
        if valid.sum() < 2:
            raise ValueError(f"Spectrum has insufficient valid samples: {spectrum.name}")
        wavelengths, reflectance = wavelengths[valid], reflectance[valid]
        order = np.argsort(wavelengths)
        wavelengths, reflectance = wavelengths[order], reflectance[order]
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError(f"Spectrum wavelengths must be strictly increasing: {spectrum.name}")
        if wavelengths[0] > grid[0] or wavelengths[-1] < grid[-1]:
            raise ValueError(
                f"Spectrum does not cover the sensor wavelength grid: {spectrum.name}"
            )
        interpolated = np.interp(grid, wavelengths, np.clip(reflectance, 0.0, 1.0))
        values = []
        for band in sensor_response.band_names:
            response = np.asarray(sensor_response.rsr[band], dtype=float)
            if response.shape != grid.shape:
                raise ValueError(
                    f"Sensor response shape does not match wavelength grid for band {band}"
                )
            denominator = np.trapezoid(response, grid)
            if denominator <= 0:
                raise ValueError(f"Sensor response has no support for band {band}")
            values.append(np.trapezoid(interpolated * response, grid) / denominator)
        columns.append(values)

    return EndmemberSet(
        A=np.asarray(columns, dtype=np.float32).T,
        names=[spectrum.name for spectrum in library],
        material_classes=[spectrum.material_class for spectrum in library],
        source_spectra=[str(spectrum.source_meta.get("path", spectrum.name)) for spectrum in library],
    )
