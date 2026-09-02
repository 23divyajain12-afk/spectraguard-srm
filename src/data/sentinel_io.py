import json
from pathlib import Path

import numpy as np

from src.core.config import RunConfig
from src.core.schemas import SentinelCube


def load_sentinel(scene_path: str, config: RunConfig) -> SentinelCube:
    """Load a SentinelCube stored in the project's portable NPZ format."""
    path = Path(scene_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sentinel scene does not exist: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("Only NPZ Sentinel scenes are supported by the current loader")

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
