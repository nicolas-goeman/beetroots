try :
    import cupy as xp
except :
    import numpy as xp
    
from typing import Union

from beetroots.sampler.utils import utils

from beetroots.sampler.proposal_mtm.abstract_proposal import ProposalDistribution
from beetroots.modelling.priors.smooth_indicator_prior import penalty_one_pix

class ProposalNeighborsAndIndicator(ProposalDistribution):
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    __slots__ = (
        "prior_indicator_lower_bounds",
        "prior_indicator_upper_bounds",
        "prior_indicator_margin_scale",
        "prior_spatial_list_edges",
        "prior_spatial_weights",
    )

    def __init__(
        self,
        lower_bounds: xp.ndarray,
        upper_bounds: xp.ndarray,
        indicator_margin_scale: float,
        list_edges: xp.ndarray,
        weights: xp.ndarray,
    ) -> None:
        r"""
        Parameters
        ----------
        lower_bounds : xp.ndarray of shape (D,)
            lower bounds of the indicator prior
        upper_bounds : xp.ndarray of shape (D,)
            upper bounds of the indicator prior
        indicator_margin_scale : float
            margin scale of the indicator prior
        list_edges : List[tuple[int, int]]
            list of edges
        weights : xp.ndarray of shape (n_edges,)
        """
        self.lower_bounds = xp.asarray(lower_bounds)
        self.upper_bounds = xp.asarray(upper_bounds)
        self.margin_scale = indicator_margin_scale
        self.list_edges = list_edges
        self.weights = weights
    
    def sample(
        self,
        Var: xp.ndarray,
        idx_pix: Union[dict, xp.ndarray],
        k_mtm,
    ) -> xp.ndarray:
        r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`). The distribution used to draw these vectors is:

        * the smooth indicator prior
        * a combination of the smooth indicator prior and of a Gaussian mixture defined with the set of all combinations of neighbors of component :math:`n`

        Parameters
        ----------
        Theta : xp.ndarray
            current iterate
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

        return utils.sample_conditional_spatial_and_indicator_prior(
            Var,
            self.list_edges,
            self.weights,
            self.lower_bounds,
            self.upper_bounds,
            self.margin_scale,
            idx_pix=idx_pix,
            k_mtm=k_mtm,
            seed=seed,
        )  # (n_pix, self.k_mtm, D)
    
    def neglog_pdf(
            self,
            candidates,
            idx_pix,
        ) -> xp.ndarray:
        n_pix = idx_pix.size
        k_mtm = candidates.shape[1]

        _pdf = penalty_one_pix(candidates[idx_pix].reshape(-1, candidates.shape[2:]),
                        self.prior_indicator_lower_bounds,
                        self.prior_indicator_upper_bounds,
                        self.prior_indicator_margin_scale)  # (n_pix * k_mtm)
        
        _pdf += utils.compute_nlpdf_spatial_prior_proposal(
            candidates,
            self.prior_spatial_list_edges,
            self.prior_spatial_weights,
            idx_pix)

        
        return _pdf.reshape((n_pix, k_mtm))
