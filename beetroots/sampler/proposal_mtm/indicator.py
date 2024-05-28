try :
    import cupy as xp
except :
    import numpy as xp

from beetroots.sampler.utils import utils

from beetroots.modelling.priors.smooth_indicator_prior import penalty_one_pix

class ProposalIndicator:
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior"""

    __slots__ = (
        "prior_indicator_lower_bounds",
        "prior_indicator_upper_bounds",
        "prior_indicator_margin_scale",
    )

    def sample(
        self,
        Var: xp.ndarray,
        idx_pix: xp.ndarray,
        k_mtm: int,
    ) -> xp.ndarray:
        seed = self.rng.integers(0, 1_000_000_000)

        n_pix = idx_pix.size
        _shape = self.prior_indicator_lower_bounds.shape

        return utils.sample_smooth_indicator(
                self.prior_indicator_lower_bounds,
                self.prior_indicator_upper_bounds,
                self.prior_indicator_margin_scale,
                size=(n_pix * k_mtm, *_shape),
                seed=seed,
            ).reshape((n_pix, k_mtm, *_shape))
    
    def neglog_pdf(
        self,
        candidates: xp.ndarray,
        idx_pix: xp.ndarray,
    ) -> xp.ndarray:
        r"""evaluates the neglog-pdf of the proposal distribution. Constant as it is the smooth indicator prio. it is not exact but it speeds up the computation.

        Parameters
        ----------
        candidates : xp.ndarray of shape (n_pix, k_mtm, D)
            the candidates
        idx_pix : xp.ndarray of shape (n_pix,)
            the indices of the pixels

        Returns
        -------
        xp.ndarray of shape (n_pix, k_mtm)
            the neglog-pdf of the candidates
        """
        n_pix = idx_pix.size
        k_mtm = candidates.shape[1]

        _pdf = penalty_one_pix(candidates[idx_pix].reshape(-1, candidates.shape[2:]),
                        self.prior_indicator_lower_bounds,
                        self.prior_indicator_upper_bounds,
                        self.prior_indicator_margin_scale)  # (n_pix * k_mtm)
        
        return _pdf.reshape((n_pix, k_mtm))




