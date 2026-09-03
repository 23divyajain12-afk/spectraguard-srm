from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.core.config import load_config
from src.data.sentinel_io import load_sentinel


def _write_band(path: Path, data: np.ndarray, transform, nodata=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="uint16",
        crs="EPSG:32643",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data.astype(np.uint16), 1)


def test_load_sentinel_safe_reads_required_bands_and_reprojects_b11(tmp_path):
    safe = tmp_path / "S2A_TEST.SAFE"
    transform_10m = from_origin(500000, 1000000, 10, 10)
    transform_20m = from_origin(500000, 1000000, 20, 20)
    for band in ("B02", "B03", "B04", "B08"):
        values = np.full((4, 4), 5000, dtype=np.uint16)
        values[0, 0] = 0
        _write_band(
            safe / "GRANULE" / "G1" / "IMG_DATA" / "R10m" / f"T_B{band[1:]}_10m.jp2",
            values,
            transform_10m,
        )
    _write_band(
        safe / "GRANULE" / "G1" / "IMG_DATA" / "R20m" / "T_B11_20m.jp2",
        np.full((2, 2), 7000, dtype=np.uint16),
        transform_20m,
    )
    (safe / "MTD_MSIL2A.xml").parent.mkdir(parents=True, exist_ok=True)
    (safe / "MTD_MSIL2A.xml").write_text(
        "<root><SENSING_TIME>2024-01-02T03:04:05Z</SENSING_TIME></root>",
        encoding="utf-8",
    )

    config = load_config(str(Path("config/default.yaml")))
    cube = load_sentinel(str(safe), config)

    assert cube.band_names == ["B02", "B03", "B04", "B08", "B11"]
    assert cube.data.shape == (5, 4, 4)
    assert cube.data.dtype == np.float32
    assert np.allclose(cube.data[:4, 1:, 1:], 0.5)
    assert np.isclose(cube.data[4, 2, 2], 0.7)
    assert cube.mask.shape == (4, 4)
    assert cube.mask.dtype == bool
    assert not cube.mask[0, 0]
    assert cube.acquisition_time == "2024-01-02T03:04:05Z"
    assert cube.crs == "EPSG:32643"
    assert cube.resolution_m == 10.0
    assert cube.meta["source_format"] == "sentinel-2-l2a-safe"
