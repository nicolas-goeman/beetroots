try :
    import cupy as xp
except :
    import numpy as xp
    
from typing import Union, Optional

from beetroots.sampler.utils import utils

from beetroots.sampler.proposal_mtm.abstract_proposal import ProposalDistribution
from beetroots.modelling.priors.smooth_indicator_prior import penalty_one_pix

class ProposalNeighbors(ProposalDistribution):
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    __slots__ = (
        "lower_bounds",
        "upper_bounds",
        "margin_scale",
        "list_edges",
        "weights",
        "std_correction",
    )

    def __init__(
        self,
        list_edges: xp.ndarray,
        weights: Optional[Union[xp.ndarray, bool]] = False,
        std_correction: Optional[float] = 1,
        **kwargs: dict,
    ) -> None:
        r"""
        Parameters
        ----------
        list_edges : List[tuple[int, int]]
            list of edges
        weights : xp.ndarray of shape (n_edges,)
        """
        self.list_edges = list_edges
        self.weights = weights
        self.std_correction = std_correction
    
    def sample(
        self,
        Var: xp.ndarray,
        idx_pix: Union[dict, xp.ndarray],
        k_mtm,
        rng: xp.random.Generator,
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
        seed = rng.integers(0, 1_000_000_000)


        if self.weights is False:
            weights = xp.ones(Var.shape[-1])
        else:
            weights = self.weights

        samples = utils.sample_conditional_spatial_prior(
            Var,
            self.list_edges,
            spatial_weights=weights,
            idx_pix=idx_pix,
            k_mtm=k_mtm,
            seed=seed,
            std_correction=self.std_correction,
        )  # (n_pix, self.k_mtm, D)

        return samples
    
    def neglog_pdf(
            self,
            candidates,
            idx_pix,
        ) -> xp.ndarray:
        n_pix = idx_pix.size
        k_mtm = candidates.shape[1]

        if self.weights is False:
            weights = xp.ones(candidates.shape[-1])
        else:
            weights = self.weights

        _pdf = utils.compute_nlpdf_spatial_prior_proposal(
            candidates,
            self.list_edges,
            weights,
            idx_pix)
        
        return _pdf.reshape((n_pix, k_mtm))
