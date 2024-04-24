import pytest
try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


from beetroots.modelling.forward_maps.basic_exp import BasicExpForwardMap
from beetroots.modelling.likelihoods import utils
from beetroots.modelling.hierarchical.aux_var_u import AuxiliaryGivenTarget


@pytest.fixture(scope="module")
def settings():
    D = 3
    L = 5
    N = 64
    return D, L, N


@pytest.fixture(scope="module")
def points(settings):
    D, L, N = settings

    x1 = xp.zeros((N, D))
    x1[-N // 2 :] += 1

    x2 = xp.ones((N, D))
    return x1, x2


@pytest.fixture(scope="module")
def my_aux_distribution(settings, points):
    D, L, N = settings
    x1, x2 = points
    forward_map = BasicExpForwardMap(D, L, N)

    sigma_a, sigma_m = 0.5, 0.1
    omega = 3 * sigma_a
    y = forward_map.evaluate(x1)  # fully uncensored
    y = xp.maximum(y, omega)

    my_aux_distribution = AuxiliaryGivenTarget(
        forward_map, D, L, N, y, sigma_a, sigma_m, omega
    )
    return my_aux_distribution


@pytest.fixture(scope="module")
def forward_map_evals_Theta1(my_aux_distribution, points):
    x1, _ = points
    forward_map_evals_Theta1 = my_aux_distribution.evaluate_all_forward_map(x1, True)
    return forward_map_evals_Theta1


@pytest.fixture(scope="module")
def forward_map_evals_Theta2(my_aux_distribution, points):
    _, x2 = points
    forward_map_evals_Theta2 = my_aux_distribution.evaluate_all_forward_map(x2, True)
    return forward_map_evals_Theta2


@pytest.fixture(scope="module")
def nll_utils_Theta1(my_aux_distribution, forward_map_evals_Theta1):
    nll_utils_Theta1 = my_aux_distribution.evaluate_all_nll_utils(forward_map_evals_Theta1)
    return nll_utils_Theta1


@pytest.fixture(scope="module")
def nll_utils_Theta2(my_aux_distribution, forward_map_evals_Theta2):
    nll_utils_Theta2 = my_aux_distribution.evaluate_all_nll_utils(forward_map_evals_Theta2)
    return nll_utils_Theta2