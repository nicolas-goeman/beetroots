from abc import ABC, abstractmethod
from typing import Optional, Union

from beetroots.modelling.component_distribution import ComponentDistribution

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class Hierarchical(ComponentDistribution):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
    ) -> None:
        super().__init__(D, L, N)

        self.hyperparameters = None

    @abstractmethod
    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(
        self,
        forward_map_evals: dict[str, xp.ndarray],
        nll_utils: dict[str, xp.ndarray],
        var_name: str,
    ) -> xp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        forward_map_evals: dict,
        nll_utils: dict,
        var_name: str,
    ) -> xp.ndarray:
        raise NotImplementedError
