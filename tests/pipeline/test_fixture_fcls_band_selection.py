from dataclasses import replace
from pathlib import Path

import numpy as np

from src.core.config import load_config
from src.pipeline.orchestrator import _load_endmembers


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "aoi_small.npz"


def test_sen2sr_fixture_endmembers_use_four_common_bands():
    config = load_config(str(ROOT / "config" / "default.yaml"))
    config = replace(
        config,
        sr=replace(config.sr, model="sen2sr"),
    )

    selected = _load_endmembers(config, FIXTURE)

    assert selected is not None
    assert selected.A.shape == (4, 3)
    with np.load(FIXTURE, allow_pickle=False) as values:
        np.testing.assert_array_equal(selected.A, values["endmember_A"][:4])
