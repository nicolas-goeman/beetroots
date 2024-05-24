from beetroots.modelling.hierarchical.abstract_hierarchical import Hierarchical
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.likelihoods import utils
from typing import Optional, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed
from scipy.special import log_ndtr


class AuxiliaryGivenTarget(Likelihood, Hierarchical):
    r"""Class implementing the conditional distribution :math:`U|\Theta` for both full conditional distributions :math:`\pi(U|\Theta, Y)` and :math:`\pi(\Theta|U, Y)`."""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        var_names: dict,
        sigma_m: Union[float, xp.ndarray],
    ) -> None:
        super().__init__(forward_map, D, L, N)

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m
        self.sigma_m2 = xp.square(self.sigma_m)

        assert isinstance(var_names, dict) and list(var_names.keys()) == ["aux", "target"]

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out = self.nlpdf_utils['log_aux']
        out += xp.square(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'])/(2*self.sigma_m2)
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def gradient_neglog_pdf_wrt_aux(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += 3/2 + (self.nlpdf_utils['log_aux'] - self.forward_map_evals['log_f_Var'])/self.sigma_m2
        out /= self.nlpdf_utils['aux']

        return out
    
    def gradient_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out -= self.forward_map_evals['grad_log_f_Var']*(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'])
        out /= self.sigma_m2

        return out
            
    def gradient_neglog_pdf(
        self,
        var_name: str,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
         
        if var_name == self.var_names["aux"]:
            out = self.gradient_neglog_pdf_wrt_aux(pixelwise, full, idx)
        elif var_name == self.var_names["target"]:
            out = self.gradient_neglog_pdf_wrt_target(pixelwise, full, idx)
        else:
            raise ValueError(f"var_name must be either {self.var_names["aux"]} or {self.var_names["target"]}")

        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def hess_diag_neglog_pdf_wrt_aux(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += 3/2 + (self.nlpdf_utils['log_aux'] - self.forward_map_evals['log_f_Var']- 1)/self.sigma_m2
        out /= -self.nlpdf_utils['aux2']

        return out

    def hess_diag_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += self.forward_map_evals['hess_diag_log_f_Var']*(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'])
        out -= self.forward_map_evals['grad_log_f_Var']**2
        out /= self.sigma_m2

        return out
    
    def hess_diag_neglog_pdf(
        self,
        var_name: str,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        
        if var_name == self.var_names["aux"]:
            out = self.hess_neglog_pdf_wrt_aux(pixelwise, full, idx)
        elif var_name == self.var_names["target"]:
            out = self.hess_neglog_pdf_wrt_target(pixelwise, full, idx)
        else:
            raise ValueError(f"var_name must be either {self.var_names["aux"]} or {self.var_names["target"]}")

        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def evaluate_all_nlpdf_utils(
        self,
        var_name: str,
        idx: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        # TODO: implement nlpdf_utils method
        self.nlpdf_utils = {}

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = self.forward_map_evals["log_f_Var"] - xp.square(self.sigma_m)/2 + self.sigma_m*rng.normal(loc=0, scale=1, size=self.forward_map_evals["log_f_Var"].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
        out = xp.exp(out)
    
        return out
    
            
class ObservationsGivenAuxiliary(Likelihood):
    r"""Class implementing the conditional distribution :math:`Y|U` for the full conditional distribution :math:`\pi(U|\Theta, Y)`."""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        y: xp.ndarray,
        sigma_a: Union[float, xp.ndarray],
        sigma_m: Union[float, xp.ndarray],
        omega: xp.ndarray,
    ) -> None:
        super().__init__(D, L, N)

        if not (y.shape == (N, L)):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        elif isinstance(y, xp.ndarray):
            self.y = y
        else:
            raise ValueError("y must be an array")
        
        if isinstance(sigma_a, (float, int)):
            self.sigma_a = sigma_a * xp.ones((N, L))
        else:
            assert sigma_a.shape == (N, L)
            self.sigma_a = sigma_a

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m

        if isinstance(omega, (float, int)):
            self.omega = omega * xp.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega
    

    def neglog_pdf_u(
        self,
        y: xp.ndarray,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += xp.square(y-self.nlpdf_utils['aux'])
        out /= 2*self.nlpdf_utils['s_a2'] 
        
        return out
    
    def neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out -= log_ndtr(z)

        return out

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["nlpdf_c"],
            self.nlpdf_utils["nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out

    def grad_neglog_pdf_u(
        self,
        y: xp.ndarray,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out -= y-self.nlpdf_utils['aux']
        out /= self.nlpdf_utils['s_a2'] 
        
        return out
    
    def grad_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out += utils.norm_pdf_cdf_ratio(z)
        out /= self.nlpdf_utils['s_a']

        return out

    def grad_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["grad_nlpdf_c"],
            self.nlpdf_utils["grad_nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def hess_diag_diag_neglog_pdf_u(
        self,
        y: xp.ndarray,
    ) -> Union[float, xp.ndarray]:
        out = xp.ones((self.N, self.L))

        out /= self.nlpdf_utils['s_a2'] 
        
        return out
    
    def hess_diag_diag_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out += xp.square(utils.norm_pdf_cdf_ratio(z))
        out += utils.norm_pdf_cdf_ratio(z)*z
        out /= self.nlpdf_utils['s_a2']

        return out

    def hess_diag_diag_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["hess_diag_diag_nlpdf_c"],
            self.nlpdf_utils["hess_diag_diag_nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        ) -> None:
        self.nlpdf_utils = {} #TODO: implement nlpdf_utils method

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = self.nlpdf_utils['aux'] + self.sigma_a*rng.normal(loc=0, scale=1, size=self.nlpdf_utils['aux'].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
    
        return out
    