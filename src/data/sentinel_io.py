import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject

from src.core.config import RunConfig
from src.core.schemas import SentinelCube


SAFE_BANDS = ("B02", "B03", "B04", "B08", "B11")


def _load_npz(path: Path) -> SentinelCube:
    with np.load(path, allow_pickle=False) as values:
        required = {
            "data", "band_names", "crs", "transform", "resolution_m",
            "bounds", "mask", "meta_json",
        }
        missing = required.difference(values.files)
        if missing:
            raise ValueError(f"Sentinel scene is missing fields: {sorted(missing)}")

        data = np.asarray(values["data"], dtype=np.float32)
        band_names = [str(name) for name in values["band_names"].tolist()]
        mask = np.asarray(values["mask"], dtype=bool)
        if data.ndim != 3 or data.shape[0] != len(band_names):
            raise ValueError("Scene data must have shape (bands, height, width)")
        if mask.shape != data.shape[1:]:
            raise ValueError("Scene mask must have shape (height, width)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Scene data contains non-finite reflectance values")

        metadata = json.loads(str(values["meta_json"].item()))
        return SentinelCube(
            data=np.clip(data, 0.0, 1.0),
            band_names=band_names,
            crs=str(values["crs"].item()),
            transform=tuple(float(value) for value in values["transform"]),
            resolution_m=float(values["resolution_m"].item()),
            bounds=tuple(float(value) for value in values["bounds"]),
            mask=mask,
            acquisition_time=(
                str(values["acquisition_time"].item())
                if "acquisition_time" in values.files
                else None
            ),
            meta=metadata,
        )


def _find_safe_band(scene_path: Path, band: str, resolution: int) -> Path:
    candidates = sorted(
        path for path in (scene_path / "GRANULE").rglob("*.jp2")
        if f"_{band}_{resolution}m" in path.name
        and "IMG_DATA" in path.parts
        and f"R{resolution}m" in path.parts
    )
    if not candidates:
        raise FileNotFoundError(
            f"Sentinel SAFE scene is missing {band} at {resolution} m"
        )
    return candidates[0]


def _read_acquisition_time(scene_path: Path) -> Optional[str]:
    metadata_files = [scene_path / "MTD_MSIL2A.xml"]
    metadata_files.extend(scene_path.rglob("MTD_TL.xml"))
    for metadata_file in metadata_files:
        if metadata_file.is_file():
            text = metadata_file.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"<SENSING_TIME>([^<]+)</SENSING_TIME>", text)
            if match:
                return match.group(1).strip()
    return None


def _read_band(
    path: Path,
    shape: Optional[Tuple[int, int]] = None,
    crs=None,
    transform=None,
    window: Optional[Window] = None,
) -> Tuple[np.ndarray, np.ndarray, object, object, object]:
    with rasterio.open(path) as source:
        read_window = window or Window(0, 0, source.width, source.height)
        values = source.read(1, window=read_window)
        valid = source.read_masks(1, window=read_window) > 0
        if source.nodata is not None:
            valid &= values != source.nodata
        if shape is None:
            return (
                values.astype(np.float32),
                valid,
                source.crs,
                source.window_transform(read_window),
                source.window_bounds(read_window),
            )

        destination = np.zeros(shape, dtype=np.float32)
        destination_mask = np.zeros(shape, dtype=np.uint8)
        source_transform = source.window_transform(read_window)
        reproject(
            values,
            destination,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=source.nodata,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
        reproject(
            valid.astype(np.uint8),
            destination_mask,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        return destination, destination_mask.astype(bool), crs, transform, None


def _load_safe(path: Path, aoi_bbox: Optional[Tuple[float, float, float, float]] = None) -> SentinelCube:
    if not path.is_dir() or path.suffix.upper() != ".SAFE":
        raise ValueError(f"Not a Sentinel-2 SAFE directory: {path}")

    band_paths: Dict[str, Path] = {
        band: _find_safe_band(path, band, 20 if band == "B11" else 10)
        for band in SAFE_BANDS
    }
    with rasterio.open(band_paths["B02"]) as reference_source:
        scene_bounds = reference_source.bounds
        if aoi_bbox is None:
            reference_window = Window(
                0, 0, reference_source.width, reference_source.height
            )
        else:
            minx, miny, maxx, maxy = aoi_bbox
            if minx >= maxx or miny >= maxy:
                raise ValueError("AOI bounding box must have positive area")
            if (
                maxx <= scene_bounds.left
                or minx >= scene_bounds.right
                or maxy <= scene_bounds.bottom
                or miny >= scene_bounds.top
            ):
                raise ValueError("AOI does not intersect the Sentinel scene")
            clipped = (
                max(minx, scene_bounds.left),
                max(miny, scene_bounds.bottom),
                min(maxx, scene_bounds.right),
                min(maxy, scene_bounds.top),
            )
            reference_window = from_bounds(*clipped, transform=reference_source.transform)
            reference_window = reference_window.round_offsets().round_lengths()

    reference, reference_mask, reference_crs, reference_transform, bounds = _read_band(
        band_paths["B02"], window=reference_window
    )
    shape = reference.shape
    data_bands = []
    valid_mask = reference_mask.copy()
    for band in SAFE_BANDS:
        if band == "B02":
            values, band_mask = reference, reference_mask
        elif band != "B11":
            values, band_mask, _, _, _ = _read_band(
                band_paths[band], window=reference_window
            )
        else:
            with rasterio.open(band_paths[band]) as source:
                b11_window = from_bounds(
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                    transform=source.transform,
                ).round_offsets().round_lengths()
            values, band_mask, _, _, _ = _read_band(
                band_paths[band],
                shape,
                reference_crs,
                reference_transform,
                window=b11_window,
            )
        data_bands.append(values / 10000.0)
        valid_mask &= band_mask

    data = np.stack(data_bands).astype(np.float32)
    data[:, ~valid_mask] = 0.0
    data[:, valid_mask] = np.clip(data[:, valid_mask], 0.0, 1.0)
    metadata = {
        "scene_id": path.name[:-5],
        "source_format": "sentinel-2-l2a-safe",
        "reflectance_scale": 10000,
        "band_sources": {
            band: str(band_paths[band].relative_to(path))
            for band in SAFE_BANDS
        },
    }
    return SentinelCube(
        data=data,
        band_names=list(SAFE_BANDS),
        crs=reference_crs.to_string(),
        transform=tuple(reference_transform)[:6],
        resolution_m=10.0,
        bounds=(
            bounds[0], bounds[1], bounds[2], bounds[3]
        ),
        mask=valid_mask.astype(bool),
        nodata=0.0,
        acquisition_time=_read_acquisition_time(path),
        meta=metadata,
    )


def load_sentinel(
    scene_path: str,
    config: RunConfig,
    aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> SentinelCube:
    """Load either the portable NPZ fixture or a Sentinel-2 L2A SAFE scene."""
    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(f"Sentinel scene does not exist: {path}")
    if path.is_dir():
        return _load_safe(path, aoi_bbox)
    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    raise ValueError("Supported Sentinel scenes are NPZ files or .SAFE directories")
