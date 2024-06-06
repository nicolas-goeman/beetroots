try :
    import cupy as xp
except :
    import numpy as xp

from beetroots.sampler.utils import utils

from beetroots.modelling.priors.smooth_indicator_prior import penalty_one_pix

class ProposalIndicator:
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior"""

    __slots__ = (
        "lower_bounds",
        "upper_bounds",
        "margin_scale",
    )

    def __init__(
        self,
        lower_bounds: xp.ndarray,
        upper_bounds: xp.ndarray,
        indicator_margin_scale: float,
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
        """
        self.lower_bounds = xp.asarray(lower_bounds)
        self.upper_bounds = xp.asarray(upper_bounds)
        self.margin_scale = indicator_margin_scale

    def sample(
        self,
        Var: xp.ndarray,
        idx_pix: xp.ndarray,
        k_mtm: int,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        seed = rng.integers(0, 1_000_000_000)

        n_pix = idx_pix.size
        _shape = self.lower_bounds.shape

        return utils.sample_smooth_indicator(
                self.lower_bounds,
                self.upper_bounds,
                self.margin_scale,
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

        _pdf = penalty_one_pix(candidates[idx_pix].reshape(-1, *candidates.shape[2:]),
                        self.lower_bounds,
                        self.upper_bounds,
                        self.margin_scale)  # (n_pix * k_mtm)
        _pdf = _pdf.sum(axis=tuple(range(1, _pdf.ndim)))
        
        return _pdf.reshape((n_pix, k_mtm))




