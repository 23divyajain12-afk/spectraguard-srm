from collections.abc import Mapping

import numpy as np

from src.core.schemas import SentinelCube
from src.data.preprocessing import _resize_band


def reproject_match(
    src: SentinelCube,
    target_crs: str,
    target_transform: tuple,
) -> SentinelCube:
    """Resample a cube to the target transform and assign its target CRS."""
    if len(target_transform) != 6:
        raise ValueError("target_transform must be an affine 6-tuple")
    pixel_width = abs(float(target_transform[0]))
    pixel_height = abs(float(target_transform[4]))
    if pixel_width == 0 or pixel_height == 0:
        raise ValueError("target_transform must contain non-zero pixel sizes")

    width = max(1, round((src.bounds[2] - src.bounds[0]) / pixel_width))
    height = max(1, round((src.bounds[3] - src.bounds[1]) / pixel_height))
    data = np.stack([_resize_band(band, height, width) for band in src.data])
    mask = _resize_band(src.mask.astype(np.float32), height, width) >= 0.5
    metadata = dict(src.meta)
    metadata["reprojected_from_crs"] = src.crs
    return SentinelCube(
        data=data,
        band_names=list(src.band_names),
        crs=target_crs,
        transform=tuple(float(value) for value in target_transform),
        resolution_m=(pixel_width + pixel_height) / 2,
        bounds=src.bounds,
        mask=mask,
        nodata=src.nodata,
        acquisition_time=src.acquisition_time,
        meta=metadata,
    )


def crop_to_aoi(cube: SentinelCube, aoi_geom) -> SentinelCube:
    """Crop a cube to an AOI bounding box in the cube's coordinate system."""
    if isinstance(aoi_geom, Mapping):
        coordinates = aoi_geom.get("coordinates")
        if aoi_geom.get("type") == "Polygon" and coordinates:
            points = coordinates[0]
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
        elif "bbox" in aoi_geom:
            min_x, min_y, max_x, max_y = aoi_geom["bbox"]
        else:
            raise ValueError("AOI mapping must be a Polygon or contain bbox")
    else:
        if len(aoi_geom) != 4:
            raise ValueError("AOI must be (minx, miny, maxx, maxy)")
        min_x, min_y, max_x, max_y = aoi_geom

    left, bottom, right, top = cube.bounds
    min_x, max_x = max(left, min_x), min(right, max_x)
    min_y, max_y = max(bottom, min_y), min(top, max_y)
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("AOI does not intersect the cube")

    pixel_width = abs(cube.transform[0])
    pixel_height = abs(cube.transform[4])
    col_start = max(0, int(np.floor((min_x - left) / pixel_width)))
    col_stop = min(cube.data.shape[2], int(np.ceil((max_x - left) / pixel_width)))
    row_start = max(0, int(np.floor((top - max_y) / pixel_height)))
    row_stop = min(cube.data.shape[1], int(np.ceil((top - min_y) / pixel_height)))
    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError("AOI produces an empty crop")

    new_bounds = (
        left + col_start * pixel_width,
        top - row_stop * pixel_height,
        left + col_stop * pixel_width,
        top - row_start * pixel_height,
    )
    transform = list(cube.transform)
    transform[2] = new_bounds[0]
    transform[5] = new_bounds[3]
    return SentinelCube(
        data=cube.data[:, row_start:row_stop, col_start:col_stop].copy(),
        band_names=list(cube.band_names),
        crs=cube.crs,
        transform=tuple(transform),
        resolution_m=cube.resolution_m,
        bounds=new_bounds,
        mask=cube.mask[row_start:row_stop, col_start:col_stop].copy(),
        nodata=cube.nodata,
        acquisition_time=cube.acquisition_time,
        meta=dict(cube.meta),
    )
