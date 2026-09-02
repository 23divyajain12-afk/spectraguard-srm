import numpy as np

from src.core.config import load_config
from src.core.schemas import EndmemberSet, FusedCube
from src.unmixing.fcls import fcls


def test_fcls_enforces_nonnegative_sum_to_one():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    data = np.array([[[0.8, 0.2]], [[0.2, 0.8]]], dtype=np.float32)
    cube = FusedCube(
        data=data,
        band_names=["B1", "B2"],
        crs="EPSG:32643",
        transform=(1, 0, 0, 0, -1, 1),
        resolution_m=1,
        bounds=(0, 0, 2, 1),
        mask=np.ones((1, 2), dtype=bool),
    )
    result = fcls(cube, EndmemberSet(matrix, ["a", "b"], ["a", "b"], ["a", "b"]), load_config("config/default.yaml"))
    assert result.S.shape == (2, 1, 2)
    assert np.all(result.S >= 0.0)
    np.testing.assert_allclose(result.S.sum(axis=0), 1.0, atol=1e-6)
    assert result.solver == "fcls"
