try :
    import cupy as xp
except :
    import numpy as xp
    
from typing import Union

from beetroots.sampler.utils import utils

from beetroots.modelling.priors.smooth_indicator_prior import penalty_one_pix

class ProposalNeighborsAndIndicator:
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    __slots__ = (
        "prior_indicator_lower_bounds",
        "prior_indicator_upper_bounds",
        "prior_indicator_margin_scale",
        "prior_spatial_list_edges",
        "prior_spatial_weights",
    )

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
            self.prior_spatial_list_edges,
            self.prior_spatial_weights,
            self.prior_indicator_lower_bounds,
            self.prior_indicator_upper_bounds,
            self.prior_indicator_margin_scale,
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
