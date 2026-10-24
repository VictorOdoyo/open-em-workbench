import numpy as np
import pytest

from emlab import uncertainty as m


def test_correlated_linear_uncertainty_and_mean_bootstrap_have_distinct_meaning():
    covariance = np.array([[4.0, 1], [1, 9]])
    jacobian = np.array([2.0, -1])
    assert m.linearized_covariance(jacobian, covariance)[0, 0] == pytest.approx(21)
    low, high = m.mean_bootstrap(np.arange(100, dtype=float), resamples=1000)
    assert low < 49.5 < high
    assert high - low < 20
    assert m.mean_bootstrap(np.full(10, 3.0), resamples=100) == (3.0, 3.0)
