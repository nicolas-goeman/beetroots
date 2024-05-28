from typing import List, Optional

import numba as nb
try:
    import cupy as xp
    decorator_nb = nb.cuda.jit
except:
    import numpy as xp
    decorator_nb = nb.njit

from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution


@decorator_nb("float64[:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    neglog_p = xp.zeros((D,))
    for d in range(D):
        neglog_p[d] = xp.sum(
            xp.where(
                Var[:, d] > upper_bounds[d],
                (Var[:, d] - upper_bounds[d]) / indicator_margin_scale,
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (lower_bounds[d] - Var[:, d]) / indicator_margin_scale,
                    0,
                ),
            )
            ** 4
        )
    return neglog_p  # (D,)


@decorator_nb()
def penalty_one_pix(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    neglog_p_full = (
        xp.where(
            Var > xp.expand_dims(upper_bounds, 0),
            (Var - xp.expand_dims(upper_bounds, 0)) / indicator_margin_scale,
            xp.where(
                Var < xp.expand_dims(lower_bounds, 0),
                (xp.expand_dims(lower_bounds, 0) - Var) / indicator_margin_scale,
                xp.zeros_like(Var),
            ),
        )
        ** 4
    )  # (N_candidates, D)
    return xp.sum(neglog_p_full, axis=1)  # (N_candidates,)


@decorator_nb("float64[:,:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def gradient_penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    g = xp.zeros_like(Var)
    for d in range(D):
        g[:, d] = (
            4
            / indicator_margin_scale**4
            * xp.where(
                Var[:, d] > upper_bounds[d],
                (Var[:, d] - upper_bounds[d]),
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Var[:, d]),
                    0,
                ),
            )
            ** 3
        )
    return g


@decorator_nb("float64[:,:](float64[:,:], float64[:], float64[:], float64)", nopython=True)
def hess_diag_penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    hess_diag = xp.zeros_like(Var)
    for d in range(D):
        hess_diag[:, d] = (
            12
            / indicator_margin_scale**4
            * xp.where(
                Var[:, d] > upper_bounds[d],
                (Var[:, d] - upper_bounds[d]),
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Var[:, d]),
                    0,
                ),
            )
            ** 2
        )
    return hess_diag  # (N, D)


class SmoothIndicatorPrior(PriorProbaDistribution):
    r"""This prior encodes validity intervals :math:`[\underline{\theta}_{d}, \overline{\theta}_{d}]]` the physical parameters :math:`\theta_{nd}` (for :math:`d \in [\![1, D]\!]`).
    The negative log of this prior is

    .. math::

        \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
        0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
        \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
        \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
        \end{cases}

    with :math:`\Delta > 0` a margin scaling parameter.
    """

    __slots__ = (
        "D",
        "N",
        "indicator_margin_scale",
        "lower_bounds",
        "upper_bounds",
    )

    def __init__(
        self,
        D: int,
        N: int,
        indicator_margin_scale: float,
        lower_bounds: xp.ndarray,
        upper_bounds: xp.ndarray,
        list_idx_sampling: List[int],
    ) -> None:
        super().__init__(D, N)
        self.indicator_margin_scale = indicator_margin_scale
        r"""float: scaling parameter :math:`\Delta`"""

        self.lower_bounds_full = lower_bounds
        r"""xp.ndarray: validity interval lower bounds of the full set of D physical parameters"""
        self.upper_bounds_full = upper_bounds
        r"""xp.ndarray: validity interval upper bounds of the full set of D physical parameters"""

        self.lower_bounds = lower_bounds[list_idx_sampling]
        r"""xp.ndarray: validity interval lower bounds of the set of sampled physical parameters"""
        self.upper_bounds = upper_bounds[list_idx_sampling]
        r"""xp.ndarray: validity interval upper bounds of the set of sampled physical parameters"""

        assert (
            self.lower_bounds.size == self.D
        ), f"should be {self.D}, is {self.lower_bounds.size}"
        assert (
            self.upper_bounds.size == self.D
        ), f"should be {self.D}, is {self.upper_bounds.size}"

        return

    def neglog_pdf(
            self,
            Var: xp.ndarray,
            idx_pix: Optional[xp.ndarray] = None,
            pixelwise: bool = False) -> xp.ndarray:
        r"""compute the negative log of the prior that approximates the indicator function

        .. math::

            \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
            0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
            \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
            \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
            \end{cases}

        Parameters
        ----------
        Var : xp.array of shape (N, D)
            current iterate
        pixelwise : bool, optional
            wether to return an aggregated result per pixel (if True) or per map (if False), by default False

        Returns
        -------
        neglog_p : xp.ndarray of shape (D,) or (N,)
            negative log of the smooth indicator prior pdf
        """
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        if pixelwise:
            neglog_p = penalty_one_pix(
                Var,
                self.lower_bounds,
                self.upper_bounds,
                self.indicator_margin_scale,
            )  # (N,)
        else:
            neglog_p = penalty(
                Var,
                self.lower_bounds,
                self.upper_bounds,
                self.indicator_margin_scale,
            )  # (D,)
        return neglog_p

    def neglog_pdf_one_pix(self, Var: xp.ndarray) -> xp.ndarray:
        r"""compute the negative log of the prior that approximates the indicator function

        .. math::

            \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
            0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
            \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
            \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
            \end{cases}

        Parameters
        ----------
        Var : xp.array of shape (N_candidates, D)
            current iterate

        Returns
        -------
        neglog_p : numpy array of shape (N_candidates,)
            negative log of the smooth indicator prior per map
        """
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        neglog_p = penalty_one_pix(
            Var,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N_candidates,)
        return neglog_p

    def gradient_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        r"""gradient of the negative log pdf of the smooth indicator prior

        Parameters
        ----------
        Var : xp.array of shape (N, D)
            current iterate

        Returns
        -------
        g : xp.array of shape (N, D)
            gradient
        """
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        grad_ = gradient_penalty(
            Var,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N, D)
        return grad_  # / (self.N * self.D)

    def hess_diag_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        r"""diagonal of the Hessian of the negative log pdf of the smooth indicator prior

        Parameters
        ----------
        Var : xp.array of shape (N, D)
            current iterate

        Returns
        -------
        hess_diag : xp.array of shape (N, D)
            [description]
        """
        assert len(Var.shape) == 2 and Var.shape[1] == self.D
        hess_diag = hess_diag_penalty(
            Var,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N, D)
        return hess_diag  # / (self.N * self.D)
