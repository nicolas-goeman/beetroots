from abc import ABC, abstractmethod
from typing import Union, Optional

from beetroots.modelling.component_distribution import ComponentDistribution

import numpy as np


class PriorProbaDistribution(ComponentDistribution):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(self, D: int, N: int) -> None:
        self.D = D
        """int: number of distinct physical parameters"""
        self.N = N
        """int: number of pixels in each physical dimension"""

    @abstractmethod
    def neglog_pdf(self, var: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(self, var: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hess_diag_neglog_pdf(self, var: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx: Optional[np.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        ) -> None:
        """Evaluate all utilities for the negative log-pdf"""
        raise NotImplementedError
    