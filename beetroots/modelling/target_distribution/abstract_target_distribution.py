from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class TargetDistribution(ABC):

    __slots__ = (
        "D",
        "L",
        "N",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        distribution_components: list,
    ):
        self.D = D
        """int: number of distinct physical parameters"""

        self.L = L
        """int: number of observables per pixel"""

        self.N = N
        """int: number of pixels"""

        self.distribution_components = distribution_components
        """list: list of all distributions that compose the target distribution"""

        return

    @abstractmethod
    def neglog_pdf(
        self,
        nlpdf_utils: dict,
        full: bool = False,
    ) -> float:
        pass

    @abstractmethod
    def grad_neglog_pdf(
        self,
        nlpdf_utils: dict,
    ) -> xp.ndarray:
        pass
    
    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        nlpdf_utils: dict,
    ) -> xp.ndarray:
        pass

    @abstractmethod
    def compute_all_for_saver(
        self,
        nlpdf_utils: dict,
    ) -> Tuple[dict[str, Union[float, xp.ndarray]], xp.ndarray]:
        """computes negative log pdf of each component distribution and posterior (detailed values to be saved, not to be used in sampling)

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : [str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_nll_utils()`` method

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            values to be saved
        """
        pass

    @abstractmethod
    def compute_all(
        self,
        nlpdf_utils,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the target distribution

        Parameters
        ----------
        nlpdf_utils : dict[str, np.ndarray]
            output of the union of the outputs of the method ``evaluate_all_nll_utils()`` of each component distribution
        compute_derivatives : bool, optional
            whether to compute 1st order derivatives, by default True
        compute_derivatives_second_order : bool, optional
            whether to compute second order derivatives, by default True

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            negative log pdf and derivatives of the posterior distribution
        """
        pass
