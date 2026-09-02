import json
from pathlib import Path
from typing import Tuple

import numpy as np

from src.core.schemas import EndmemberSet, SentinelCube


def load_small_aoi(
    path: str | Path | None = None,
) -> Tuple[SentinelCube, EndmemberSet]:
    """Load the deterministic synthetic Sentinel cube and toy endmembers."""
    fixture_path = Path(path) if path is not None else Path(__file__).with_name("aoi_small.npz")

    with np.load(fixture_path, allow_pickle=False) as values:
        metadata = json.loads(str(values["meta_json"].item()))
        cube = SentinelCube(
            data=values["data"],
            band_names=values["band_names"].tolist(),
            crs=str(values["crs"].item()),
            transform=tuple(float(value) for value in values["transform"]),
            resolution_m=float(values["resolution_m"].item()),
            bounds=tuple(float(value) for value in values["bounds"]),
            mask=values["mask"],
            nodata=None,
            acquisition_time=str(values["acquisition_time"].item()),
            meta=metadata,
        )
        endmembers = EndmemberSet(
            A=values["endmember_A"],
            names=values["endmember_names"].tolist(),
            material_classes=values["material_classes"].tolist(),
            source_spectra=values["source_spectra"].tolist(),
        )

    return cube, endmembers
