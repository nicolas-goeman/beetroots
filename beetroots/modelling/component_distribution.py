from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class ComponentDistribution(ABC):
    r"""Abstract Base Class for a probability distribution on non-countable set"""

    def __init__(
        self,
        #vars_involved: Optional[Union[str, Tuple[str]]],
        **kwargs,
    ) -> None:
        self.nlpdf_utils = {}

        # self.vars_involved = set(vars_involved)

    @abstractmethod
    def neglog_pdf(
        self,
        full: bool = False,
        **kwargs,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def gradient_neglog_pdf(
        self,
        **kwargs,
    ) -> xp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        **kwargs,
    ) -> xp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        **kwargs,
        ) -> None:
        """Evaluate all utilities for the negative log-pdf and its eventual derivatives"""
        raise NotImplementedError