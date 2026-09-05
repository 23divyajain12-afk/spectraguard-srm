import logging

import numpy as np
import pytest

from src.pipeline.orchestrator import _log_stage
from src.sr.tiling import tiled_inference


class UpscaleModel:
    def predict(self, tile):
        array = np.asarray(tile)
        return np.repeat(np.repeat(array, 2, axis=-2), 2, axis=-1)


def test_tiled_progress_reports_dynamic_completed_tiles():
    updates = []
    image = np.ones((3, 10, 13), dtype=np.float32)

    tiled_inference(
        UpscaleModel(),
        image,
        tile_size=6,
        overlap=2,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert updates[-1] == (12, 12)
    assert [completed for completed, _ in updates] == list(range(1, 13))


def test_tiled_progress_does_not_report_failed_tile():
    class FailingModel:
        def predict(self, tile):
            raise RuntimeError("inference failed")

    updates = []
    with pytest.raises(RuntimeError, match="inference failed"):
        tiled_inference(
            FailingModel(),
            np.ones((3, 4, 4), dtype=np.float32),
            tile_size=4,
            overlap=0,
            progress_callback=lambda completed, total: updates.append(
                (completed, total)
            ),
        )

    assert updates == []


def test_stage_progress_uses_stage_percentages(caplog):
    with caplog.at_level(logging.INFO):
        _log_stage(1, "Loading Sentinel-2")
        _log_stage(9, "FCLS unmixing")

    assert "[ 11%] [1/9] Loading Sentinel-2..." in caplog.text
    assert "[100%] [9/9] FCLS unmixing..." in caplog.text
