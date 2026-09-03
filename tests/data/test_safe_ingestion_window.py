from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.core.config import load_config
from src.data.sentinel_io import load_sentinel


def _write(path: Path, values: np.ndarray, transform, resolution):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="uint16", crs="EPSG:32643", transform=transform, nodata=0,
    ) as dataset:
        dataset.write(values, 1)


def test_safe_aoi_reads_only_requested_grid(tmp_path):
    safe = tmp_path / "WINDOW.SAFE"
    t10 = from_origin(500000, 1000000, 10, 10)
    t20 = from_origin(500000, 1000000, 20, 20)
    for band in ("B02", "B03", "B04", "B08"):
        _write(
            safe / "GRANULE" / "G" / "IMG_DATA" / "R10m" / f"T_{band}_10m.jp2",
            np.full((10, 10), 4000, dtype=np.uint16),
            t10,
            10,
        )
    _write(
        safe / "GRANULE" / "G" / "IMG_DATA" / "R20m" / "T_B11_20m.jp2",
        np.full((5, 5), 8000, dtype=np.uint16),
        t20,
        20,
    )

    config = load_config(str(Path("config/default.yaml")))
    cube = load_sentinel(str(safe), config, aoi_bbox=(500020, 999920, 500060, 999960))

    assert cube.data.shape == (5, 4, 4)
    assert cube.bounds == (500020.0, 999920.0, 500060.0, 999960.0)
    assert cube.transform[2] == 500020.0
    assert cube.transform[5] == 999960.0
    assert np.all(cube.mask)
    assert np.allclose(cube.data[:4], 0.4)
    assert np.allclose(cube.data[4], 0.8)


def test_safe_aoi_rejects_non_intersecting_bbox(tmp_path):
    safe = tmp_path / "EMPTY.SAFE"
    safe.mkdir()
    config = load_config(str(Path("config/default.yaml")))

    # The loader validates the AOI against the reference scene after band discovery.
    for band in ("B02", "B03", "B04", "B08"):
        _write(
            safe / "GRANULE" / "G" / "IMG_DATA" / "R10m" / f"T_{band}_10m.jp2",
            np.ones((2, 2), dtype=np.uint16),
            from_origin(0, 20, 10, 10),
            10,
        )
    _write(
        safe / "GRANULE" / "G" / "IMG_DATA" / "R20m" / "T_B11_20m.jp2",
        np.ones((1, 1), dtype=np.uint16),
        from_origin(0, 20, 20, 20),
        20,
    )

    import pytest
    with pytest.raises(ValueError, match="does not intersect"):
        load_sentinel(str(safe), config, aoi_bbox=(100, 100, 120, 120))
