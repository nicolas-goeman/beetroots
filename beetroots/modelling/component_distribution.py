from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class ComponentDistribution(ABC):
    @abstractmethod
    def neglog_pdf(
        self,
        nlpdf_utils: dict,
        full: bool = False,
        **kwargs,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad_neglog_pdf(
        self,
        nlpdf_utils: dict,
        **kwargs,
    ) -> xp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        nlpdf_utils: dict,
        **kwargs,
    ) -> xp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_all_nlpdf_utils(
        self,
        var: xp.ndarray,
    ) -> dict:
        """Evaluate all utilities for the negative log-pdf"""
        raise NotImplementedError