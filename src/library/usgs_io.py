from pathlib import Path

import numpy as np

from src.core.schemas import RawSpectrum


def load_usgs_spectrum(path: str) -> RawSpectrum:
    """Load a documented text spectrum with wavelength and reflectance columns."""
    spectrum_path = Path(path)
    if not spectrum_path.is_file():
        raise FileNotFoundError(f"Spectrum file does not exist: {spectrum_path}")

    rows = []
    name = spectrum_path.stem
    material_class = "unknown"
    for line in spectrum_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if ":" in stripped:
                key, value = stripped[1:].split(":", 1)
                if key.strip().lower() == "name":
                    name = value.strip()
                elif key.strip().lower() in {"material_class", "class"}:
                    material_class = value.strip()
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError("Spectrum file must contain at least two numeric rows")

    values = np.asarray(rows, dtype=np.float32)
    order = np.argsort(values[:, 0])
    wavelengths = values[order, 0]
    reflectance = values[order, 1]
    valid = np.isfinite(wavelengths) & np.isfinite(reflectance)
    if valid.sum() < 2 or np.any(np.diff(wavelengths[valid]) <= 0):
        raise ValueError("Spectrum wavelengths must be finite and strictly increasing")
    return RawSpectrum(
        name=name,
        material_class=material_class,
        wavelength_nm=wavelengths[valid],
        reflectance=np.clip(reflectance[valid], 0.0, 1.0),
        source_meta={"path": str(spectrum_path), "format": "two-column text"},
    )
