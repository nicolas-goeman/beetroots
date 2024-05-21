try :
    import cupy as xp
except :
    import numpy as xp

from beetroots.modelling.target_distribution import TargetDistribution
from beetroots.sampler.utils import utils



from typing import List, Union

import numpy as np


class MTMRandomStartParams:
    r"""Dataclass to contain data on the spatial prior"""

    __slots__ = (
        "name",
        "use_next_nearest_neighbors",
        "initial_regu_weights",
    )

    def __init__(
        self,
        name: str,
        use_next_nearest_neighbors: bool,
        initial_regu_weights: Union[np.ndarray, List[float]],
    ) -> None:
        r"""

        Parameters
        ----------
        name : str
            name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]
        use_next_nearest_neighbors : bool
            wether or not to use the next nearest neighbors, i.e., in diagonal
        initial_regu_weights : Union[np.ndarray, List[float]]
            initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)
        """
        assert name in ["L2-laplacian", "L2-gradient"]

        self.name = name
        r"""str: name of the spatial regularization type, must be an element of ["L2-laplacian", "L2-gradient"]"""

        self.use_next_nearest_neighbors = use_next_nearest_neighbors
        r"""bool: wether or not to use the next nearest neighbors, i.e., in diagonal"""

        self.initial_regu_weights = initial_regu_weights
        r"""Union[np.ndarray, List[float]: initial regularization weights (the regularization weights can be tuned automatically during the Markov chain)"""


def generate_random_start_Theta_1pix(
    self, Theta: xp.ndarray, target_distribution: TargetDistribution, idx_pix: xp.ndarray
) -> xp.ndarray:
    r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`). The distribution used to draw these vectors is:

    * the smooth indicator prior
    * a combination of the smooth indicator prior and of a Gaussian mixture defined with the set of all combinations of neighbors of component :math:`n`

    Parameters
    ----------
    Theta : xp.ndarray
        current iterate
    target_distribution : TargetDistribution
        contains the lower and upper bounds of the hypercube
    idx_pix : xp.ndarray
        indices of the pixels

    Returns
    -------
    xp.array of shape (n_pix, self.k_mtm, D)
        random element of the hypercube defined by the lower and upper bounds with uniform distribution

    Raises
    ------
    ValueError : if ``target_distribution.prior_indicator`` is None
    """
    seed = self.rng.integers(0, 1_000_000_000)
    n_pix = idx_pix.size

    if not hasattr(target_distribution, 'prior_indicator') or target_distribution.prior_indicator is None:
        raise ValueError("The target distribution has no specified smooth indicator prior")

    if not hasattr(target_distribution, 'prior_spatial') or target_distribution.prior_spatial is None:
        # * sample from smooth indicator prior
        return utils.sample_smooth_indicator(
            target_distribution.prior_indicator.lower_bounds,
            target_distribution.prior_indicator.upper_bounds,
            target_distribution.prior_indicator.indicator_margin_scale,
            size=(n_pix * self.k_mtm, self.D),
            seed=seed,
        ).reshape((n_pix, self.k_mtm, self.D))

    else:
        return utils.sample_conditional_spatial_and_indicator_prior(
            Theta,
            target_distribution.prior_spatial.list_edges,
            target_distribution.prior_spatial.weights,
            target_distribution.prior_indicator.lower_bounds,
            target_distribution.prior_indicator.upper_bounds,
            target_distribution.prior_indicator.indicator_margin_scale,
            idx_pix=idx_pix,
            k_mtm=self.k_mtm,
            seed=seed,
        )  # (n_pix, self.k_mtm, D)