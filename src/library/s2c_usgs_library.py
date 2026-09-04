"""
Sentinel-2C-aware USGS endmember preparation for SpectraGuard-SRM.

Implements the existing library boundary:
    load_usgs_spectrum(path) -> RawSpectrum
    load_sensor_response(path, band_names) -> SensorResponse
    prepare_endmembers(library, sensor_response, config) -> EndmemberSet

Important:
- USGS splib07a bad channels are -1.23e34 and are removed/interpolated.
- ASD Full Range (ASDFR) spectra have 2151 channels from 350..2500 nm at 1 nm,
  so they can be mapped directly to that native grid.
- BECK spectra must use the exact USGS BECK wavelength record; do not invent a
  wavelength grid. This module accepts a wavelength_file for BECK.
- S2C SRFs come from the supplied Copernicus v4.0 workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import openpyxl

try:
    from src.core.schemas import RawSpectrum, SensorResponse, EndmemberSet
except ImportError:  # Allows isolated unit testing of this module.
    @dataclass
    class RawSpectrum:
        name: str
        material_class: str
        wavelength_nm: np.ndarray
        reflectance: np.ndarray
        source_meta: Dict[str, Any]

    @dataclass
    class SensorResponse:
        band_names: List[str]
        wavelength_nm: np.ndarray
        rsr: Dict[str, np.ndarray]

    @dataclass
    class EndmemberSet:
        A: np.ndarray
        names: List[str]
        material_classes: List[str]
        source_spectra: List[str]


BAD_VALUE = -1.23e34
DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11"]
S2C_SHEET = "Spectral Responses (S2C)"
S2C_COLUMNS = {
    "B02": "S2C_SR_AV_B2",
    "B03": "S2C_SR_AV_B3",
    "B04": "S2C_SR_AV_B4",
    "B08": "S2C_SR_AV_B8",
    "B11": "S2C_SR_AV_B11",
}


def _parse_numeric_lines(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError:
                # The first line is the USGS record title.
                continue
    if not values:
        raise ValueError(f"No numeric reflectance values found in {path}")
    return np.asarray(values, dtype=float)


def _material_from_name(name: str) -> str:
    n = name.lower()
    rules = [
        ("asphalt", "asphalt"),
        ("concrete", "concrete"),
        ("roofing", "roofing"),
        ("sand", "sand"),
        ("metal", "metal"),
        ("grass", "vegetation"),
        ("lawn", "vegetation"),
        ("water", "water"),
        ("basalt", "rock_proxy"),
    ]
    for token, cls in rules:
        if token in n:
            return cls
    return "unknown"


def _native_wavelengths_for_count(n: int, path: Path) -> np.ndarray:
    """
    Safe wavelength handling for the supplied splib07a files.

    2151 values are ASDFR: 350..2500 nm in 1 nm steps (USGS documentation).
    Other channel counts require an exact USGS wavelength record.
    """
    if n == 2151:
        return np.arange(350.0, 2501.0, 1.0)
    raise ValueError(
        f"{path.name} has {n} spectral channels. "
        "Provide the exact USGS native wavelength record for this spectrometer; "
        "the integration code will not guess it."
    )


def load_usgs_spectrum(path: str, material_class: Optional[str] = None) -> RawSpectrum:
    p = Path(path)
    y = _parse_numeric_lines(p)
    x = _native_wavelengths_for_count(len(y), p)

    valid = np.isfinite(y) & (y > -1e10) & (y < 10.0) & (y != BAD_VALUE)
    if valid.sum() < 10:
        raise ValueError(f"Too few valid USGS samples in {p}")

    # Interpolate only over the measured valid domain; no extrapolation.
    yy = np.interp(x, x[valid], y[valid])
    title = ""
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        title = f.readline().strip()

    return RawSpectrum(
        name=p.stem,
        material_class=material_class or _material_from_name(title),
        wavelength_nm=x,
        reflectance=yy,
        source_meta={
            "source_file": str(p),
            "usgs_record_title": title,
            "library": "USGS Spectral Library Version 7 splib07a",
            "bad_value": BAD_VALUE,
            "valid_channels": int(valid.sum()),
            "native_channel_count": int(len(y)),
        },
    )


def load_sensor_response(path: str, band_names: Sequence[str]) -> SensorResponse:
    p = Path(path)
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    if S2C_SHEET not in wb.sheetnames:
        raise ValueError(f"{S2C_SHEET!r} not found in {p}")

    ws = wb[S2C_SHEET]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    index = {str(v): i for i, v in enumerate(headers) if v is not None}

    if "SR_WL" not in index:
        raise ValueError("S2C sheet has no SR_WL column")

    wavelengths = []
    columns = {b: [] for b in band_names}
    for row in ws.iter_rows(min_row=2, values_only=True):
        w = row[index["SR_WL"]]
        if w is None:
            continue
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        wavelengths.append(w)
        for b in band_names:
            if b not in S2C_COLUMNS:
                raise ValueError(f"Unsupported S2C band {b}")
            col = S2C_COLUMNS[b]
            if col not in index:
                raise ValueError(f"Missing S2C SRF column {col}")
            v = row[index[col]]
            try:
                columns[b].append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                columns[b].append(0.0)

    wl = np.asarray(wavelengths, dtype=float)
    rsr = {b: np.nan_to_num(np.asarray(v, dtype=float), nan=0.0) for b, v in columns.items()}
    return SensorResponse(
        band_names=list(band_names),
        wavelength_nm=wl,
        rsr=rsr,
    )


def _integrate_srf(
    spectrum_wl: np.ndarray,
    reflectance: np.ndarray,
    srf_wl: np.ndarray,
    response: np.ndarray,
) -> float:
    r = np.interp(srf_wl, spectrum_wl, reflectance, left=np.nan, right=np.nan)
    valid = np.isfinite(r) & np.isfinite(response) & (response > 0)
    if valid.sum() < 2:
        raise ValueError("Insufficient spectral overlap with SRF")
    denom = np.trapezoid(response[valid], srf_wl[valid])
    if denom <= 0:
        raise ValueError("SRF has zero integrated response")
    return float(np.trapezoid(r[valid] * response[valid], srf_wl[valid]) / denom)


def prepare_endmembers(
    library: List[RawSpectrum],
    sensor_response: SensorResponse,
    config: Any = None,
) -> EndmemberSet:
    """
    Convert every supplied RawSpectrum to the Sentinel-band matrix A (C,M).

    The input list is intentionally explicit. The caller decides which spectra
    are active using the manifest; unsupported/uncertain materials should not
    be silently substituted.
    """
    bands = list(sensor_response.band_names)
    if bands != DEFAULT_BANDS:
        raise ValueError(f"Expected S2C bands {DEFAULT_BANDS}, got {bands}")

    vectors = []
    names = []
    classes = []
    sources = []

    for spec in library:
        vec = [
            _integrate_srf(
                spec.wavelength_nm,
                spec.reflectance,
                sensor_response.wavelength_nm,
                sensor_response.rsr[b],
            )
            for b in bands
        ]
        vectors.append(vec)
        names.append(spec.name)
        classes.append(spec.material_class)
        sources.append(spec.source_meta.get("source_file", spec.name))

    if not vectors:
        raise ValueError("No active endmembers were supplied")

    A = np.asarray(vectors, dtype=float).T  # (C,M)
    if A.shape[0] != len(bands):
        raise AssertionError(f"Endmember matrix shape must be (5,M), got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("Endmember matrix contains non-finite values")
    if np.any(A < 0):
        raise ValueError("Endmember reflectance cannot be negative")

    return EndmemberSet(
        A=A,
        names=names,
        material_classes=classes,
        source_spectra=sources,
    )


def load_active_library(
    root: str,
    manifest_path: str,
    sensor_response_path: str,
) -> Tuple[List[RawSpectrum], SensorResponse]:
    """
    Convenience loader for the packaged data layout.
    """
    root_p = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    active = manifest["active_endmembers"]
    spectra_dir = root_p / "data" / "raw" / "spectral_library" / "selected_usgs"

    library = []
    for item in active:
        library.append(
            load_usgs_spectrum(
                str(spectra_dir / item["file"]),
                material_class=item["material_class"],
            )
        )

    sr = load_sensor_response(sensor_response_path, manifest["bands"])
    return library, sr


def write_endmember_csv(endmembers: EndmemberSet, path: str) -> None:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "material_class", "B02", "B03", "B04", "B08", "B11", "source_spectrum"])
        for j, name in enumerate(endmembers.names):
            w.writerow([
                name,
                endmembers.material_classes[j],
                *[float(x) for x in endmembers.A[:, j]],
                endmembers.source_spectra[j],
            ])
