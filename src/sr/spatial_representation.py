import numpy as np

from src.core.config import RunConfig
from src.core.schemas import SentinelCube, SpatialRepresentation


def build_spatial_input(
    cube: SentinelCube, config: RunConfig
) -> SpatialRepresentation:
    """Build a normalized pseudo-RGB or single-band SR input."""
    if cube.data.ndim != 3 or cube.data.shape[0] == 0:
        raise ValueError("cube.data must have shape (bands, height, width)")

    if config.sr.model == "sen2sr":
        requested = ["B02", "B03", "B04", "B08"]
    else:
        requested = list(config.data.bands)
    if requested:
        indices = []
        for band in requested:
            if band not in cube.band_names:
                raise ValueError(f"Configured band is not present in cube: {band}")
            indices.append(cube.band_names.index(band))
    else:
        indices = list(range(min(3, cube.data.shape[0])))

    if not indices:
        raise ValueError("At least one source band is required")
    array = cube.data[indices].astype(np.float32, copy=True)
    valid = cube.mask
    if not np.any(valid):
        raise ValueError("Cannot normalize a cube with no valid pixels")

    minimum = array[:, valid].min(axis=1)
    maximum = array[:, valid].max(axis=1)
    scale = np.where(maximum > minimum, maximum - minimum, 1.0)
    array = np.clip(
        (array - minimum[:, None, None]) / scale[:, None, None],
        0.0,
        1.0,
    )
    normalization = {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }
    method = "pseudo_rgb" if len(indices) in (3, 4) else "single_band"
    return SpatialRepresentation(
        array=array[0] if len(indices) == 1 else array,
        method=method,
        source_bands=[cube.band_names[index] for index in indices],
        normalization=normalization,
    )
