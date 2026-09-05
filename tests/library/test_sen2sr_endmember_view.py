import numpy as np

from src.core.schemas import EndmemberSet
from src.library.endmember_preparation import (
    SEN2SR_BANDS,
    select_endmember_bands,
)
from src.unmixing.fcls import fcls
from tests.unmixing.test_fcls_adversarial import _config
from tests.unmixing.test_fcls_adversarial import _cube


ALL_BANDS = ["B02", "B03", "B04", "B08", "B11"]


def _five_band_endmembers() -> EndmemberSet:
    return EndmemberSet(
        A=np.asarray(
            [
                [0.8, 0.1],
                [0.7, 0.2],
                [0.6, 0.3],
                [0.5, 0.4],
                [0.4, 0.5],
            ],
            dtype=np.float32,
        ),
        names=["material_a", "material_b"],
        material_classes=["a", "b"],
        source_spectra=["a.txt", "b.txt"],
    )


def test_five_band_endmembers_remain_unchanged():
    endmembers = _five_band_endmembers()
    selected = select_endmember_bands(endmembers, ALL_BANDS, ALL_BANDS)

    np.testing.assert_array_equal(selected.A, endmembers.A)
    assert selected.A.shape == (5, 2)


def test_sen2sr_view_selects_ordered_four_common_bands_without_b11():
    endmembers = _five_band_endmembers()
    selected = select_endmember_bands(endmembers, ALL_BANDS)

    assert SEN2SR_BANDS == ["B02", "B03", "B04", "B08"]
    assert selected.A.shape == (4, 2)
    np.testing.assert_array_equal(selected.A, endmembers.A[:4])


def test_four_band_view_supports_fcls_constraints():
    endmembers = _five_band_endmembers()
    selected = select_endmember_bands(endmembers, ALL_BANDS)
    observations = np.asarray([[0.71, 0.19, 0.59, 0.49]], dtype=np.float32)

    result = fcls(_cube(observations), selected, _config())

    assert np.all(result.S >= 0.0)
    np.testing.assert_allclose(result.S.sum(axis=0), 1.0, atol=1e-6)
